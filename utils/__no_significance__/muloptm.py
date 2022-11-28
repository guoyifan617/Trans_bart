#encoding: utf-8

import torch

from utils.dynbatch import softmax
from utils.torch.comp import torch_no_grad

def getlr(optml):

	lr = []
	for optm in optml:
		for i, param_group in enumerate(optm.param_groups):
			lr.append(float(param_group["lr"]))

	return lr

def backup_para(plin):

	with torch_no_grad():
		rs = [pu.data.clone() for pu in plin]

	return rs

def restore_para(plin, rlin, copy=True):

	with torch_no_grad():
		if copy:
			for pu, ru in zip(plin, rlin):
				pu.data.copy_(ru.data)
		else:
			for pu, ru in zip(plin, rlin):
				pu.data = ru.data

def back_restore_para(plin, rlin, copy=True):

	rs = []
	with torch_no_grad():
		if copy:
			for pu, ru in zip(plin, rlin):
				rs.append(pu.data)
				pu.data = ru.data.clone().detach()
		else:
			for pu, ru in zip(plin, rlin):
				rs.append(pu.data)
				pu.data = ru.data

	return rs

def wacc_para(pl, wl):

	with torch_no_grad():
		w, nd = torch.as_tensor(wl, device=pl[0][0].device, dtype=pl[0][0].dtype), len(wl)
		rs = [torch.stack(plu, -1).view(-1, nd).mv(w).view(plu[0].size()) for plu in zip(*pl)]

	return rs

def optm_step(model, optm, lossf, evaf, batch_data, base_loss):

	_mpl, loptmind = backup_para(model.parameters()), len(optm) - 1
	plind = loptmind - 1
	_aop = []
	_aodl = []
	for i, _optm in enumerate(optm):
		_optm.step()
		_aodl.append(base_loss - evaf(model, lossf, batch_data))
		_aop.append(back_restore_para(model.parameters(), _mpl, copy=(i < plind)) if i < loptmind else list(model.parameters()))
	_aodl = softmax(_aodl)
	restore_para(model.parameters(), wacc_para(_aop, _aodl), copy=False)

	return _aodl

def report_dl(dl, ndl):

	if ndl > 0:
		_tmp = float(ndl) / 100.0

		return " ".join(["%.2f" % (du / _tmp,) for du in dl])
	else:
		return "None"
