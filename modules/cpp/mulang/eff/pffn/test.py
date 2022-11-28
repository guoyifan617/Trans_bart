#encoding: utf-8

"""
this script reports wired results on 2 Titan X:
20
warm up devices
00:20, 48.70it/s
py for loop
01:12, 138.46it/s
c openmp
01:17, 129.31it/s
py threads
01:17, 129.31it/s
single device baseline
01:12, 137.67it/s
"""

import torch
from threading import Lock, Thread
from torch import nn
from torch.utils.cpp_extension import load

from modules.base import PositionwiseFF as PositionwiseFFBase
from utils.torch.comp import torch_autocast, torch_is_autocast_enabled, torch_is_grad_enabled, torch_set_grad_enabled
from utils.torch.ext import ensure_num_threads
from utils.tqdm import tqdm

from cnfg.ihyp import *

def parallel_apply(modules, inputs, devices, kwargs_tup=None, lock=None):

	if kwargs_tup is None:
		kwargs_tup = ({},) * len(modules)

	lock = Lock() if lock is None else lock
	results = {}
	grad_enabled, torch_autocast_enabled = torch_is_grad_enabled(), torch_is_autocast_enabled()

	def _worker(i, module, input, kwargs, device=None):

		if not isinstance(input, (list, tuple,)):
			input = (input,)
		with torch_set_grad_enabled(grad_enabled), torch.cuda.device(device), torch_autocast(enabled=torch_autocast_enabled):
			output = module(*input, **kwargs)
		with lock:
			results[i] = output

	threads = [Thread(target=_worker, args=(i, module, input, kwargs, device)) for i, (module, input, kwargs, device) in enumerate(zip(modules, inputs, kwargs_tup, devices))]

	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	outputs = []
	for i in range(len(inputs)):
		output = results[i]
		outputs.append(output)

	return outputs

class PositionwiseFF(nn.Module):

	def __init__(self, isize, hsize=None, dropout=0.0, norm_residual=norm_residual_default, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, devices=None, **kwargs):

		super(PositionwiseFF, self).__init__()

		_hsize = isize * 4 if hsize is None else hsize

		self.nets = nn.ModuleList([PositionwiseFFBase(isize, hsize=_hsize, dropout=dropout, norm_residual=norm_residual, custom_act=custom_act, enable_bias=enable_bias) for i in range(len(devices))])
		for net, device in zip(self.nets, devices):
			net.to(device, non_blocking=True)
		self.devices = devices
		self.lock = Lock()

		if self.c_available():
			self.c_init()

	def forward(self, xl, **kwargs):

		return [net(x) for net, x in zip(self.nets, xl)]

	def sig_forward(self, xl):

		return [self.nets[0](xl[0])]

	def pyp_forward(self, xl):

		return parallel_apply(self.nets, xl, self.devices, lock=self.lock)

	def c_available(self):

		return type(self) == PositionwiseFF and all(net.c_available() for net in self.nets)

	def c_init(self, bind=bind_c_forward):

		try:
			import mulang_eff_ppff_cpp
		except Exception as e:
			mulang_eff_ppff_cpp = load(name="mulang_eff_ppff_cpp", sources=["modules/cpp/mulang/eff/pffn/ppff.cpp", "modules/cpp/mulang/eff/pffn/ppff_func.cpp", "modules/cpp/base/ffn/pff_func.cpp", "modules/cpp/act/act_func.cpp"], extra_cflags=["-fopenmp"])
		self.c_forward_func = mulang_eff_ppff_cpp.forward
		for net in self.nets:
			net.c_init()
		self.c_build_cache()
		if bind:
			PositionwiseFF.forward = PositionwiseFF.c_forward

	def c_forward(self, xl):

		return self.c_forward_func(*self.c_build_inputs(xl))

	def c_build_cache(self):

		self.aargs = self.nets[0].aargs

	def c_build_inputs(self, xl):

		_i_d, bargs = self.nets[0].c_build_inputs(xl[0])[:2]
		i_d = [_i_d]
		i_d.extend([net.c_build_inputs(x)[0] for net, x in zip(self.nets[1:], xl[1:])])

		return i_d, bargs, *self.aargs

ndevices = 2
print(ensure_num_threads(ndevices))
devices = [torch.device("cuda", index=i) for i in range(ndevices)]
bsize, seql, isize = 96, 64, 512
niter = 200
tmod = PositionwiseFF(isize, dropout=0.1, devices=devices)
td = [torch.randn(bsize, seql, isize, device=device) for device in devices]
print("warm up devices")
for i in tqdm(range(niter // 10)):
	rs = tmod(td)
	rs = tmod.c_forward(td)
	rs = tmod.pyp_forward(td)
print("py for loop")
for i in tqdm(range(niter)):
	rs = tmod(td)
print("c openmp")
for i in tqdm(range(niter)):
	rs = tmod.c_forward(td)
print("py threads")
for i in tqdm(range(niter)):
	rs = tmod.pyp_forward(td)
print("single device baseline")
for i in tqdm(range(niter)):
	rs = tmod.sig_forward(td)
