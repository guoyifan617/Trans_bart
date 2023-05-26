#encoding: utf-8

import torch
from time import time

from parallel.parallelMT import DataParallelMT
from transformer.MuLang.NMT import NMT
from utils.base import set_random_seed
from utils.fmt.base4torch import parse_cuda_decode
from utils.h5serial import h5File
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode
from utils.tqdm import tqdm

import cnfg.mulang as cnfg
from cnfg.ihyp import *

warm_up = 3
niter = 20

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
use_amp = cnfg.use_amp and use_cuda

set_random_seed(cnfg.seed, use_cuda)

td = h5File(cnfg.dev_data, "r")

ntest = td["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, ntask, nwordt = nword[0], nword[1], nword[-1]

mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, ntask=ntask, ngroup=cnfg.ngroup)

mymodel.apply(load_fixing)

mymodel.eval()

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	if multi_gpu:
		mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)

beam_size = cnfg.beam_size
length_penalty = cnfg.length_penalty

src_grp, tgt_grp, task_grp = td["src"], td["tgt"], td["task"]

for i in range(warm_up):
	with torch_inference_mode():
		for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_t = torch.from_numpy(task_grp[bid][()])
			decode_len = torch.from_numpy(tgt_grp[bid][()]).size(1)
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device, non_blocking=True)
				seq_t = seq_t.to(cuda_device, non_blocking=True)
			seq_batch, seq_t = seq_batch.long(), seq_t.long()
			with torch_autocast(enabled=use_amp):
				output = mymodel.decode(seq_batch, seq_t, beam_size, decode_len, length_penalty)

start = time()
for i in range(niter):
	with torch_inference_mode():
		for i in tqdm(range(ntest), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_t = torch.from_numpy(task_grp[bid][()])
			decode_len = torch.from_numpy(tgt_grp[bid][()]).size(1)
			if cuda_device:
				seq_batch = seq_batch.to(cuda_device, non_blocking=True)
				seq_t = seq_t.to(cuda_device, non_blocking=True)
			seq_batch, seq_t = seq_batch.long(), seq_t.long()
			with torch_autocast(enabled=use_amp):
				output = mymodel.decode(seq_batch, seq_t, beam_size, decode_len, length_penalty)
cost = time() - start

td.close()
print(cost)
