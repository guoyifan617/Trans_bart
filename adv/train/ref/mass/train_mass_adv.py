#encoding: utf-8

import torch
from math import floor
from random import shuffle
from torch.nn.parallel.scatter_gather import gather
from torch.optim import Adam as Optimizer

from loss.base import MultiLabelSmoothingLoss
#from lrsch import GoogleLR as LRScheduler
from parallel.base import DataParallelCriterion
from parallel.parallelMono import DataParallelMT
from transformer.Mono.AdvNMT import NMT
from transformer.Mono.NMT import NMT as BaseNMT
from utils.base import free_cache, get_logger, mkdir, set_random_seed
from utils.fmt.base import iter_to_str
from utils.fmt.base4torch import load_emb, parse_cuda
from utils.fmt.mono.base import init_token_id, mask_id
from utils.h5serial import h5File
from utils.init.base import init_model_params
from utils.io import load_model_cpu, save_model, save_states
from utils.mask.base import update_p
from utils.mask.mass import get_batch
from utils.state.holder import Holder
from utils.state.pyrand import PyRandomState
from utils.state.thrand import THRandomState
from utils.torch.comp import torch_compile, torch_inference_mode
from utils.tqdm import tqdm
from utils.train.base import getlr, optm_step_zero_grad_set_none
from utils.train.dss import dynamic_sample

import cnfg.mono as cnfg
from cnfg.ihyp import *
from cnfg.vocab.mono import pad_id

def train(td, tl, ed, nd, optm, lrsch, model, lossf, mv_device, logger, done_tokens, multi_gpu, tokens_optm=32768, nreport=None, save_every=None, chkpf=None, state_holder=None, statesf=None, num_checkpoint=1, cur_checkid=0, report_eva=True, remain_steps=None, save_loss=False, save_checkp_epoch=False, use_amp=False, train_adv=False, iter_steps_remain=20, adv_steps=20, mt_steps=20):

	sum_loss = 0.0
	sum_wd = 0
	part_loss = 0.0
	part_wd = 0
	sum_loss_adv = 0.0
	sum_wd_adv = 0
	part_loss_adv = 0.0
	part_wd_adv = 0
	_done_tokens = done_tokens
	model.train()
	cur_b = 1
	ndata = len(tl)
	_cur_checkid = cur_checkid
	_cur_rstep = remain_steps
	cur_b, _ls = 1, {} if save_loss else None

	_train_adv = train_adv
	_iter_steps_remain = iter_steps_remain
	global minloss, wkdir, mask_ratio, random_ratio, len_ratio, nwordi, nwordt, init_token_id, mask_id
	_optm_mt, _optm_adv = optm

	_td = (td[0]["src"], td[1]["src"],)
	for i_d, t_d in tqdm(tl, mininterval=tqdm_mininterval):
		seq_batch = torch.from_numpy(_td[t_d][i_d][()])
		if _train_adv:
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
			seq_batch = seq_batch.long()
			output, loss_adv = model(seq_batch, None, lang_id=t_d, psind=None)
			loss = None
		else:
			_src_mask = seq_batch.eq(pad_id).unsqueeze(1)
			seq_batch, seq_o, _sind_o = get_batch(seq_batch, len_ratio, mask_ratio, random_ratio, mask_id, init_token_id, nwordi if t_d == 0 else nwordt)
			lo = seq_o.size(1) - 1
			if mv_device:
				seq_batch = seq_batch.to(mv_device, non_blocking=True)
				seq_o = seq_o.to(mv_device, non_blocking=True)
				_src_mask = _src_mask.to(mv_device, non_blocking=True)

			oi = seq_o.narrow(1, 0, lo)
			ot = seq_o.narrow(1, 1, lo).contiguous()
			output, loss_adv = model(seq_batch, oi, mask=_src_mask, lang_id=t_d, psind=_sind_o)
			loss = lossf(output, ot, lang_id=t_d)
		if multi_gpu:
			if loss is not None:
				loss = loss.sum()
			loss_adv = gather(loss_adv, mv_device, 0).sum()
		loss_add = 0.0 if loss is None else loss.data.item()
		loss = loss_adv if loss is None else (loss + loss_adv)

		if use_amp:
			with amp.scale_loss(loss, optm) as scaled_loss:
				scaled_loss.backward()
		else:
			loss.backward()

		_wd_count = seq_batch if _train_adv else ot
		wd_add = _wd_count.ne(0).int().sum().item()
		if _train_adv:
			loss_add_adv = loss_adv.data.item()
			sum_loss_adv += loss_add_adv
			sum_wd_adv += wd_add
		else:
			_src_mask = seq_o = oi = ot = None
			if save_loss:
				_ls[(i_d, t_d,)] = loss_add / wd_add
			sum_loss += loss_add
			sum_wd += wd_add
		loss = loss_adv = output = seq_batch = None
		_done_tokens += wd_add

		if _done_tokens >= tokens_optm:
			if _train_adv:
				_optm = _optm_adv
			else:
				#lrsch.step()
				_optm = _optm_mt
			if multi_gpu:
				model.collect_gradients()
				_optm.step()
				_optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)
				if _train_adv:
					model.module.fix_update()
				model.update_replicas()
			else:
				_optm.step()
				_optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)
				if _train_adv:
					model.fix_update()
			_iter_steps_remain -= 1
			if _iter_steps_remain == 0:
				if _train_adv:
					_train_adv = False
					model.train_advers(False)
					_iter_steps_remain = mt_steps
				else:
					_train_adv = True
					model.train_advers(True)
					_iter_steps_remain = adv_steps
			_done_tokens = 0
			if _cur_rstep is not None:
				if save_checkp_epoch and (save_every is not None) and (_cur_rstep % save_every == 0) and (chkpf is not None) and (_cur_rstep > 0):
					if num_checkpoint > 1:
						_fend = "_%d.h5" % (_cur_checkid)
						_chkpf = chkpf[:-3] + _fend
						_cur_checkid = (_cur_checkid + 1) % num_checkpoint
					else:
						_chkpf = chkpf
					save_model(model, _chkpf, multi_gpu, print_func=logger.info)
					if statesf is not None:
						save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
				_cur_rstep -= 1
				if _cur_rstep <= 0:
					break

		if nreport is not None:
			if _train_adv:
				part_loss_adv += loss_add_adv
				part_wd_adv += wd_add
			else:
				part_loss += loss_add
				part_wd += wd_add
			if cur_b % nreport == 0:
				if report_eva:
					_leva, _eeva = eva(ed, nd, model, lossf, mv_device, multi_gpu)
					logger.info("Average loss over %d tokens: %.3f %.3f, valid loss/error: %.3f %.2f" % (part_wd, part_loss / (1.0 if part_wd == 0 else part_wd), part_loss_adv / (1.0 if part_wd_adv == 0 else part_wd_adv), _leva, _eeva))
					if _leva <= minloss:
						save_model(model, wkdir + "best.h5", multi_gpu, print_func=logger.info)
						minloss = _leva
					free_cache(mv_device)
					model.train()
				else:
					logger.info("Average loss over %d tokens: %.3f" % (part_wd, part_loss / (1.0 if part_wd == 0 else part_wd)))
				part_loss = 0.0
				part_wd = 0
				part_loss_adv = 0.0
				part_wd_adv = 0

		if save_checkp_epoch and (_cur_rstep is None) and (save_every is not None) and (cur_b % save_every == 0) and (chkpf is not None) and (cur_b < ntrain):
			if num_checkpoint > 1:
				_fend = "_%d.h5" % (_cur_checkid)
				_chkpf = chkpf[:-3] + _fend
				_cur_checkid = (_cur_checkid + 1) % num_checkpoint
			else:
				_chkpf = chkpf
			save_model(model, _chkpf, multi_gpu, print_func=logger.info)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": _cur_rstep, "checkpoint_id": _cur_checkid, "training_list": tl[cur_b - 1:]}), statesf, print_func=logger.info)
		cur_b += 1
	if part_wd != 0.0:
		logger.info("Average loss over %d tokens: %.3f %.3f" % (part_wd, part_loss / (1.0 if part_wd == 0.0 else part_wd), part_loss_adv / (1.0 if part_wd_adv == 0.0 else part_wd_adv)))
	return sum_loss / (1.0 if sum_wd == 0 else sum_wd), sum_loss_adv / (1.0 if sum_wd_adv == 0 else sum_wd_adv), _done_tokens, _cur_checkid, _cur_rstep, _ls, _train_adv, _iter_steps_remain

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
			output = model(seq_batch, seq_o.narrow(1, 0, lo), lang_id=1)
			loss = lossf(output, ot, lang_id=1)
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

td = (h5File(cnfg.train_data_src, "r"), h5File(cnfg.train_data_tgt, "r"),)
vd = h5File(cnfg.dev_data, "r")

ntrain_src = td[0]["ndata"][()].item()
ntrain_tgt = td[1]["ndata"][()].item()
nvalid = vd["ndata"][()].item()
nwordi, nwordt = td[0]["nword"][()].tolist()[0], td[1]["nword"][()].tolist()[0]

tl_src = [(str(i), 0) for i in range(ntrain_src)]
tl_tgt = [(str(i), 1) for i in range(ntrain_tgt)]
sample_target = cnfg.sample_target
nsample = float(ntrain_tgt) * cnfg.sample_ratio if sample_target else float(ntrain_src) * cnfg.sample_ratio
nsample = max(1, floor(nsample))

logger.info("Design models with seed: %d" % torch.initial_seed())
mymodel = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes, adv_weight=cnfg.adv_weight)

fine_tune_m = cnfg.fine_tune_m

mymodel = init_model_params(mymodel)
mymodel.apply(init_fixing)
if fine_tune_m is not None:
	logger.info("Load pre-trained model from: " + fine_tune_m)
	_tmpm = BaseNMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
	_tmpm = load_model_cpu(fine_tune_m, _tmpm)
	_tmpm.apply(load_fixing)
	mymodel.load_base(_tmpm)
	_tmpm = None

lossf = MultiLabelSmoothingLoss(nwordt, cnfg.label_smoothing, ignore_index=pad_id, reduction="sum", forbidden_index=cnfg.forbidden_indexes)

if cnfg.src_emb is not None:
	logger.info("Load source embedding from: " + cnfg.src_emb)
	load_emb(cnfg.src_emb, mymodel.enc.wemb.weight, nwordi, cnfg.scale_down_emb, cnfg.freeze_srcemb)
if cnfg.tgt_emb is not None:
	logger.info("Load target embedding from: " + cnfg.tgt_emb)
	load_emb(cnfg.tgt_emb, mymodel.dec.wemb.weight, nwordt, cnfg.scale_down_emb, cnfg.freeze_tgtemb)

if cuda_device:
	mymodel.to(cuda_device, non_blocking=True)
	lossf.to(cuda_device, non_blocking=True)

optimizer = (Optimizer(mymodel.mt_parameters(), lr=init_lr, betas=adam_betas_default, eps=ieps_adam_default, weight_decay=cnfg.weight_decay, amsgrad=use_ams), optim.RMSprop(mymodel.adv_parameters(), lr=cnfg.adv_lr),)
for optm in optimizer:
	optm.zero_grad(set_to_none=optm_step_zero_grad_set_none)

if use_amp:
	mymodel, optimizer = amp.initialize(mymodel, optimizer, opt_level=cnfg.amp_opt)

if multi_gpu:
	mymodel = DataParallelMT(mymodel, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
	lossf = DataParallelCriterion(lossf, device_ids=cuda_devices, output_device=cuda_device.index, replicate_once=True)

train_adv = cnfg.start_with_adv
mymodel.train_advers(train_adv)
mt_steps, adv_steps = cnfg.mt_steps, cnfg.adv_steps
iter_remain_steps = (adv_steps + cnfg.adv_add_init_steps_ext) if train_adv else mt_steps
mask_ratio, random_ratio, len_ratio = cnfg.mask_ratio, cnfg.random_ratio, cnfg.len_ratio
mask_ratio, random_ratio = update_p(mask_ratio, random_ratio)

lrsch = None#LRScheduler(optimizer[0], cnfg.isize, cnfg.warm_step)

state_holder = None if statesf is None and cnt_states is None else Holder(**{"optm": optimizer[0], "lrsch": lrsch, "pyrand": PyRandomState(), "thrand": THRandomState(use_cuda=use_cuda)})

num_checkpoint = cnfg.num_checkpoint
cur_checkid = 0

tminerr = inf_default

minloss, minerr = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
logger.info("".join(("Init lr: ", ",".join(iter_to_str(getlr(optimizer[0]))), ", Dev Loss/Error: %.3f %.2f" % (minloss, minerr))))

if fine_tune_m is None:
	save_model(mymodel, wkdir + "init.h5", multi_gpu, print_func=logger.info)
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
		tminerr, adv_loss, done_tokens, cur_checkid, remain_steps, _, train_adv, iter_remain_steps = train(td, _ctl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, False, False, use_amp, train_adv, iter_remain_steps, adv_steps, mt_steps)
		vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
		logger.info("Epoch: 0, train loss: %.3f %.3f, valid loss/error: %.3f %.2f" % (tminerr, adv_loss, vloss, vprec))
		save_model(mymodel, wkdir + "train_0_%.3f_%.3f_%.2f.h5" % (tminerr, vloss, vprec), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None)
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
	if sample_target:
		shuffle(tl_tgt)
		tl = tl_src + tl_tgt[:nsample]
	else:
		shuffle(tl_src)
		tl = tl_src[:nsample] + tl_tgt
	shuffle(tl)
	free_cache(use_cuda)
	terr, adv_loss, done_tokens, cur_checkid, remain_steps, _Dws, train_adv, iter_remain_steps = train(td, tl, vd, nvalid, optimizer, lrsch, mymodel, lossf, cuda_device, logger, done_tokens, multi_gpu, tokens_optm, batch_report, save_every, chkpf, state_holder, statesf, num_checkpoint, cur_checkid, report_eva, remain_steps, dss_ws > 0, i >= start_chkp_save, use_amp, train_adv, iter_remain_steps, adv_steps, mt_steps)
	vloss, vprec = eva(vd, nvalid, mymodel, lossf, cuda_device, multi_gpu)
	logger.info("Epoch: %d, train loss: %.3f %.3f, valid loss/error: %.3f %.2f" % (i, terr, adv_loss, vloss, vprec))

	if (vprec <= minerr) or (vloss <= minloss):
		save_model(mymodel, wkdir + "eva_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu, print_func=logger.info, mtyp="eva" if save_auto_clean else None)
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
			save_model(mymodel, wkdir + "train_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu, print_func=logger.info, mtyp=("eva" if overwrite_eva else "train") if save_auto_clean else None)
			if statesf is not None:
				save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
		elif epoch_save:
			save_model(mymodel, wkdir + "epoch_%d_%.3f_%.3f_%.2f.h5" % (i, terr, vloss, vprec), multi_gpu, print_func=logger.info)

		namin += 1
		if namin >= earlystop:
			if done_tokens > 0:
				if multi_gpu:
					mymodel.collect_gradients()
				optimizer[1].step() if train_adv else optimizer[0].step()
				model.module.fix_update() if multi_gpu else model.fix_update()
				done_tokens = 0
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
	optimizer[1].step() if train_adv else optimizer[0].step()
	model.module.fix_update() if multi_gpu else model.fix_update()

save_model(mymodel, wkdir + "last.h5", multi_gpu, print_func=logger.info)
if statesf is not None:
	save_states(state_holder.state_dict(update=False, **{"remain_steps": remain_steps, "checkpoint_id": cur_checkid}), statesf, print_func=logger.info)
logger.info("model saved")

td[0].close()
td[1].close()
vd.close()
