#encoding: utf-8

import torch
from math import sqrt
from random import shuffle
from torch.optim import Adam as Optimizer

from loss.base import LabelSmoothingLoss
from lrsch import GoogleLR as LRScheduler
from parallel.base import DataParallelCriterion
from parallel.parallelNAS import DataParallelMT
from transformer.NAS.ENMT import NMT
from utils.base import free_cache, get_logger, mkdir, set_random_seed
from utils.fmt.base import iter_to_str
from utils.fmt.base4torch import load_emb, parse_cuda
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model, save_states
from utils.state.holder import Holder
from utils.state.pyrand import PyRandomState
from utils.state.thrand import THRandomState
from utils.torch.comp import torch_compile, torch_inference_mode
from utils.tqdm import tqdm
from utils.train.base import getlr, optm_step_zero_grad_set_none
from utils.train.dss import dynamic_sample

import cnfg.nas as cnfg
from cnfg.ihyp import *

class geTau:

	def __init__(self, k=256, **kwargs):

		self.k = float(k)
		self.cur_step = 0

	def __call__(self):

		self.cur_step += 1

		return max(min(1.0 / sqrt(self.cur_step / self.k), 1.0), 0.05)

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, state_holder=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False, use_amp=False, lrsch_arch=None, train_arch=False, nas_steps_remain=0, arch_design=None, arch_stable=0, arch_steps=100, model_steps=500, arch_early_stop=128, cost_weight=0.5, tausch=None, start_nas_step=8000):

	sum_loss = 0.0
	sum_wd = 0
	part_loss = 0.0
	part_wd = 0
	_done_tokens = done_tokens
	model.train()
	cur_b = 1
	ndata = len(tl)
	_cur_checkid = cur_checkid
	_cur_rstep = remain_steps
	_ls = {} if save_loss else None
	_train_arch = train_arch
	if _train_arch:
		_arch_steps = nas_steps_remain
		_model_steps = 0
	else:
		_model_steps = nas_steps_remain
		_arch_steps = 0
	model.train_arch(_train_arch)
	_arch_trained = _train_arch
	_arch_design = arch_design
	_arch_stable = arch_stable
	_arch_changed = False

	_fair_base = torch.arange(8, dtype=torch.long, device=mv_device, requires_grad=False).view(1, 1, 8)

	src_grp, tgt_grp = td["src"], td["tgt"]
	for i_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(src_grp[i_d][()])
		seq_o = torch.from_numpy(tgt_grp[i_d][()])
		lo = seq_o.size(1) - 1
		if mv_device:
			seq_batch = seq_batch.to(mv_device, non_blocking=True)
			seq_o = seq_o.to(mv_device, non_blocking=True)
		seq_batch, seq_o = seq_batch.long(), seq_o.long()

		oi = seq_o.narrow(1, 0, lo)
		ot = seq_o.narrow(1, 1, lo).contiguous()

		if _cur_rstep > start_nas_step:
			_node_mask = None
			loss_add = 0.0
			for i in range(8):
				output, sel = model(seq_batch, oi, node_mask=_node_mask)
				loss = lossf(output, ot)
				if multi_gpu:
					loss = loss.sum()
				loss /= 8.0
				loss_add += loss.data.item()

				if use_amp:
					with amp.scale_loss(loss, optm) as scaled_loss:
						scaled_loss.backward()
				else:
					loss.backward()
				loss = output = None
				with torch_inference_mode():
					_t_mask = _fair_base.ne(sel.unsqueeze(-1))
					_node_mask = _t_mask if _node_mask is None else (_node_mask + _t_mask).gt(0)
		else:
			if _train_arch:
				output, sel, arch_loss = model(seq_batch, oi)
			else:
				output, sel = model(seq_batch, oi)
			loss = lossf(output, ot)
			if multi_gpu:
				loss = loss.sum()
				if _train_arch and (arch_loss is not None):
					arch_loss = torch.cat([au.to(mv_device, non_blocking=True) for au in arch_loss], -1).sum()
			if _train_arch and (arch_loss is not None):
				loss = loss + arch_loss * cost_weight
			loss_add = loss.data.item()

			if use_amp:
				with amp.scale_loss(loss, optm) as scaled_loss:
					scaled_loss.backward()
			else:
				loss.backward()
			loss = output = None

		wd_add = ot.ne(0).int().sum().item()
		oi = ot = seq_batch = seq_o = None
		sum_loss += loss_add
		if save_loss:
			_ls[i_d] = loss_add / wd_add
		sum_wd += wd_add
		_done_tokens += wd_add

		if _done_tokens >= tokens_optm:
			lrsch_arch.step() if _train_arch else lrsch.step()
			if multi_gpu:
				model.collect_gradients()
				optm.step()
				optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)
				model.update_replicas()
			else:
				optm.step()
				optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)
			if _train_arch:
				_arch_steps -= 1
				model.set_tau(tausch())
				if _arch_steps <= 0:
					_cur_arch = model.get_design()
					if _arch_design is not None and _cur_arch == _arch_design:
						_arch_stable += 1
						if _arch_stable >= arch_early_stop:
							logger.info("Model architecture fixed for %d training steps" % (arch_early_stop * arch_steps,))
							logger.info(_cur_arch)
							_cur_rstep = 0
							break
					else:
						_arch_design = _cur_arch
						_arch_stable = 0
						_arch_changed = True
					_model_steps = model_steps
					model.train_arch(False)
					_train_arch = False
					_arch_trained = True
			else:
				if _cur_rstep <= start_nas_step:
					# restart learning rate scheduler
					if _cur_rstep == start_nas_step:
						logger.info("Allow training architecture parameters")
					_model_steps -= 1
					if _model_steps <= 0:
						_arch_steps = arch_steps
						model.train_arch(True)
						_train_arch = True
			_done_tokens = 0
			if _cur_rstep is not None:
				if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
					if num_checkpoint > 1:
						_fend = "_%d.h5" % (_cur_checkid)
						_chkpf = chkpf[:-3] + _fend
						_cur_checkid = (_cur_checkid + 1) % num_checkpoint
					else:
						_chkpf = chkpf
					save_model(model, _chkpf, multi_gpu)
					if statesf is not None:
						save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
				_cur_rstep -= 1
				if _cur_rstep <= 0:
					break

		if nreport is not None:
			part_loss += loss_add
			part_wd += wd_add
			if cur_b % nreport == 0:
				if report_eva:
					_leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu)
					logger.info("Average loss over %d tokens: %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / part_wd, _leva, _eeva))
					free_cache(mv_device)
					model.train()
				else:
					logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
				if _arch_trained and _arch_changed:
					logger.info(_arch_design)
					_arch_trained = False
					_arch_changed = False
				part_loss = 0.0
				part_wd = 0

		if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ntrain):
			if num_checkpoint > 1:
				_fend = "_%d.h5" % (_cur_checkid)
				_chkpf = chkpf[:-3] + _fend
				_cur_checkid = (_cur_checkid + 1) % num_checkpoint
			else:
				_chkpf = chkpf
			save_model(model, _chkpf, multi_gpu)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
		cur_b += 1
	if part_wd != 0.0:
		logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / part_wd))
	return sum_loss / sum_wd, _done_tokens, _cur_checkid, _cur_rstep, _ls, _train_arch, _arch_steps if _train_arch else _model_steps, _arch_design, _arch_stable

def eva(ed, nd, model, lossf, mv_device, multi_gpu):
	r = 0
	w = 0
	sum_loss = 0.0
	model.eval()
	src_grp, tgt_grp = ed["src"], ed["tgt"]
	with torch_inference_mode():
		for i in tqdm(range(nd), mininterval=tqdm_mininterval):
			bid = str(i)
			seq_batch = torch.from_numpy(src_grp[bid][()])
			seq_o = torch.from_numpy(tgt_grp[bid][()])
			lo = seq_o.size(1) - 1
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				seq_o = seq_o.to(mv_device, non_blocking=True)
			seq_batch, seq_o = seq_batch.long(), seq_o.long()
			ot = seq_o.narrow(1, 1, lo).contiguous()
			output = model(seq_batch, seq_o.narrow(1, 0, lo))
			loss = lossf(output, ot)
			if multi_gpu:
				loss = loss.sum()
				trans = torch.cat([outu.argmax(-1).to(mv_device, non_blocking=True) for outu in output], 0)
			else:
				trans = output.argmax(-1)
			sum_loss += loss.data.item()
			data_mask = ot.ne(0)
			correct = (trans.eq(ot) & data_mask).int()
			w += data_mask.int().sum().item()
			r += correct.sum().item()
			correct = data_mask = trans = loss = output = ot = seq_batch = seq_o = None
	w = float(w)
	return sum_loss / w, (w - r) / w * 100.0

def init_fixing(module):

	if hasattr(module, "fix_init"):
		module.fix_init()

def load_fixing(module):

	if hasattr(module, "fix_load"):
		module.fix_load()

rid = cnfg.run_id
earlystop = cnfg.earlystop
maxrun = cnfg.maxrun
tokens_optm = cnfg.tokens_optm
done_tokens = 0
batch_report = cnfg.batch_report
report_eva = cnfg.report_eva
use_ams = cnfg.use_ams
cnt_states = cnfg.train_statesf
save_auto_clean = cnfg.save_auto_clean
overwrite_eva = cnfg.overwrite_eva
save_every = cnfg.save_every
start_chkp_save = cnfg.epoch_start_checkpoint_save
epoch_save = cnfg.epoch_save
remain_steps = cnfg.training_steps

wkdir = "".join((cnfg.exp_dir, cnfg.data_id, "/", cnfg.group_id, "/", rid, "/"))
mkdir(wkdir)

chkpf = None
statesf = None
if save_every is not None:
	chkpf = wkdir + "checkpoint.h5"
if cnfg.save_train_state:
	statesf = wkdir + "train.states.t7"

logger = get_logger(wkdir + "train.log")

use_cuda = cnfg.use_cuda
gpuid = cnfg.gpuid

use_cuda, cuda_device, cuda_devices, multi_gpu = parse_cuda(cnfg.use_cuda, cnfg.gpuid)

if use_cuda and cnfg.amp_opt:
	try:

		from apex import amp

		use_amp = True
	except Exception as e:
		logger.info(str(e))
		use_amp = False
else:
	use_amp = False

set_random_seed(cnfg.seed, use_cuda)

td = h5File(cnfg.train_data, "r")
vd = h5File(cnfg.dev_data, "r")

ntrain = td["ndata"][()].item()
nvalid = vd["ndata"][()].item()
nword = td["nword"][()].tolist()
nwordi, nwordt = nword[0], nword[-1]

tl = [str(i) for i in range(ntrain)]

logger.info("Design models with seed: %d" % torch.initial_seed())
mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

fine_tune_m = cnfg.fine_tune_m

if fine_tune_m is None:
	mymodel = init_model_params(mymodel)
	mymodel.apply(init_fixing)
else:
	logger.info("Load pre-trained model from: " + fine_tune_m)
	mymodel = load_model_cpu(fine_tune_m, mymodel)
	mymodel.apply(load_fixing)

lossf = LabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

if cnfg.src_emb is not None:
	logger.info("Load source embedding from: " + cnfg.src_emb)
	load_emb(cnfg.src_emb, mymodel.enc.wemb.weight, nwordi, cnfg.scale_down_emb, cnfg.freeze_srcemb)
if cnfg.tgt_emb is not None:
	logger.info("Load target embedding from: " + cnfg.tgt_emb)
	load_emb(cnfg.tgt_emb, mymodel.dec.wemb.weight, nwordt, cnfg.scale_down_emb, cnfg.freeze_tgtemb)

train_arch = False
arch_steps = cnfg.arch_steps
model_steps = cnfg.model_steps
nas_steps_remain = arch_steps if train_arch else model_steps
arch_design = mymodel.get_design()
arch_stable = 0
arch_early_stop = cnfg.arch_early_stop
arch_cost_weight = cnfg.arch_cost_weight
start_nas_step = remain_steps - cnfg.warm_model_steps

mymodel.set_tau(0.5)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)

# lr will be over written by LRScheduler before used
optimizer = Optimizer(mymodel.parameters(), lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams)

if use_amp:
	mymodel, optimizer = amp.initialize(mymodel, optimizer, opt_level=cnfg.amp_opt)

mymodel.train_arch(train_arch)

if multi_gpu:
	mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
	lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

_warm_lr = int(cnfg.warm_step / (arch_steps + model_steps) * model_steps)
_warm_arch = cnfg.warm_step - _warm_lr
lrsch = LRScheduler(optimizer, cnfg.isize, _warm_lr)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)
lrsch_arch = LRScheduler(optimizer, cnfg.isize, _warm_arch)

mymodel = torch_compile(mymodel, *torch_compile_args, **torch_compile_kwargs)
lossf = torch_compile(lossf, *torch_compile_args, **torch_compile_kwargs)

state_holder = None if statesf is None and cnt_states is None else Holder(**{"optm": optimizer, "lrsch": lrsch, "pyrand": PyRandomState(), "thrand": THRandomState(use_cuda=use_cuda)})

tausch = geTau(max(1, _warm_arch // 4) if cnfg.tau_step is None else cnfg.tau_step)

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0

tminerr = inf_default

minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
logger.info("".join(("Init lr: ", ",".join(iter_to_str(getlr(optimizer))), ", Dev Loss/Error: %.3f %.2f" % (minloss, minerr))))
logger.info("Init architecture:")
logger.info(arch_design)

if fine_tune_m is None:
	save_model(mymodel, wkdir + "init.h5", multi_gpu)
	logger.info("Initial model saved")
else:
	if cnt_states is not None:
		logger.info("Loading training states")
		_remain_states = state_holder.load_state_dict(torch.load(cnt_states))
		remain_steps, cur_checkid = _remain_states["remain_steps"], _remain_states["checkpoint_id"]
		if "training_list" in _remain_states:
			_ctl = _remain_states["training_list"]
		else:
			shuffle(tl)
			_ctl = tl
		tminerr, done_tokens, cur_checkid, remain_steps, _, train_arch, nas_steps_remain, arch_design, arch_stable = train(td, torch.load(cnt_states), vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False, use_amp, lrsch_arch, train_arch, nas_steps_remain, arch_design, arch_stable, arch_steps, model_steps, arch_early_stop, arch_cost_weight, tausch, start_nas_step)
		vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
		logger.info("Epoch: 0, train loss: %.3f, valid loss/error: %.3f %.2f" % (tminerr, vloss, vprec))
		logger.info(arch_design)
		save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.h5" % (tminerr, vloss, vprec), multi_gpu)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

if cnfg.dss_ws is not None and cnfg.dss_ws > 0.0 and cnfg.dss_ws < 1.0:
	dss_ws = int(cnfg.dss_ws * ntrain)
	_Dws = {}
	_prev_Dws = {}
	_crit_inc = {}
	if cnfg.dss_rm is not None and cnfg.dss_rm > 0.0 and cnfg.dss_rm < 1.0:
		dss_rm = int(cnfg.dss_rm * ntrain * (1.0 - cnfg.dss_ws))
	else:
		dss_rm = 0
else:
	dss_ws = 0
	dss_rm = 0
	_Dws = None

namin = 0

for i in range(1, maxrun + 1):
	shuffle(tl)
	free_cache(use_cuda)
	terr, done_tokens, cur_checkid, remain_steps, _Dws, train_arch, nas_steps_remain, arch_design, arch_stable = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save, use_amp, lrsch_arch, train_arch, nas_steps_remain, arch_design, arch_stable, arch_steps, model_steps, arch_early_stop, arch_cost_weight, tausch, start_nas_step)
	vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
	logger.info("Epoch: %d, train loss: %.3f, valid loss/error: %.3f %.2f" % (i, terr, vloss, vprec))
	logger.info(arch_design)

	if (vprec <= minerr) or (vloss <= minloss):
		save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu)
		if statesf is not None:
			save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		logger.info("New best model saved")

		namin = 0

		if vprec < minerr:
			minerr = vprec
		if vloss < minloss:
			minloss = vloss

	else:
		if terr < tminerr:
			tminerr = terr
			save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		elif epoch_save:
			save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu)

		namin += 1
		if namin >= earlystop:
			if done_tokens > 0:
				if multi_gpu:
					mymodel.collect_gradients()
				optimizer.step()
				#lrsch.step()
				done_tokens = 0
				#optimizer.zero_grad(set_to_none=optm_step_zero_grad_set_none)
			logger.info("early stop")
			break

	if remain_steps is not None and remain_steps <= 0:
		logger.info("Last training step reached")
		break

	if dss_ws > 0:
		if _prev_Dws:
			for _key, _value in _Dws.items():
				if _key in _prev_Dws:
					_ploss = _prev_Dws[_key]
					_crit_inc[_key] = (_ploss - _value) / _ploss
			tl = dynamic_sample(_crit_inc, dss_ws, dss_rm)
		_prev_Dws = _Dws

if done_tokens > 0:
	if multi_gpu:
		mymodel.collect_gradients()
	lrsch_arch.step() if train_arch else lrsch.step()
	optimizer.step()

save_model(mymodel, wkdir + "last.h5", multi_gpu)
if statesf is not None:
	save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
logger.info("model saved")

td.close()
vd.close()
