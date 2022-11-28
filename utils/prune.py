#encoding: utf-8

import torch
from math import ceil

from modules.prune.bern import BernoulliParameter, CrossAttn, Embedding, LayerNorm, Linear, LinearBn, PositionwiseFF, SelfAttn
from utils.init.base import init_model_params_glorot, init_model_params_kaiming
from utils.torch.comp import torch_no_grad

def init_model_bernparams(modin, scale_glorot=None, scale_kaiming=None):

	_tmpm = init_model_params_kaiming(modin, scale_kaiming)

	with torch_no_grad():
		for _m in _tmpm.modules():
			if isinstance(_m, Embedding):
				init_model_params_glorot(_m, scale_glorot)
				if _m.padding_idx is not None:
					_m.weight.data[_m.padding_idx].zero_()
			elif isinstance(_m, (Linear, LinearBn,)):
				if _m.bias is not None:
					_m.bias.data.zero_()
			elif isinstance(_m, LayerNorm):
				_m.weight.data.fill_(1.0)
				_m.bias.data.zero_()

	return _tmpm

def sort_mask_para(modin):

	rs = torch.cat([mp.view(-1) for mp in modin.bernMaskParameters()], dim=0)

	return rs.sort(descending=False)[0]

def bern_thres(sortensor, ratio_keep):

	_nd = sortensor.numel()
	_nk = min(ceil(_nd * ratio_keep), _nd - 1)

	return sortensor[_nk].item()

def prune_bern(modin, thres=0.0):

	for _m in modin.modules():
		if isinstance(_m, BernoulliParameter):
			_m.prune(thres)

	return modin

def report_type_prune_ratio(modin, typl):

	rsp = rsa = 0
	for _m in modin.modules():
		if isinstance(_m, typl):
			for _tmpm in _m.modules():
				if isinstance(_tmpm, BernoulliParameter):
					_tmpd = _tmpm.data
					rsp += _tmpd.eq(0.0).int().sum().item()
					rsa += _tmpd.numel()

	return float(rsp) / float(rsa) if rsa != 0 else 0.0

def report_prune_ratio(modin):

	srcd = {"Embedding":Embedding, "MHAttn":(SelfAttn, CrossAttn,), "FFN":PositionwiseFF}
	rsd = {}
	for k, v in srcd.items():
		rsd[k] = report_type_prune_ratio(modin, v)

	return rsd

def force_mask_data(modin):

	with torch_no_grad():
		for _m in modin.modules():
			if isinstance(_m, BernoulliParameter) and (_m.fixed_mask is not None):
				_m.data.data.masked_fill_(_m.fixed_mask, 0.0)

	return modin

def force_mask_grad(modin):

	with torch_no_grad():
		for _m in modin.modules():
			if isinstance(_m, BernoulliParameter) and (_m.fixed_mask is not None):
				_m.data.grad.masked_fill_(_m.fixed_mask, 0.0)

	return modin

def remove_maskq(modin):

	for _m in modin.modules():
		if isinstance(_m, BernoulliParameter):
			_m.maskp = None

	return modin

def gen_mask(modin):

	with torch_no_grad():
		for _m in modin.modules():
			if isinstance(_m, BernoulliParameter):
				_m.fixed_mask = _m.data.eq(0.0)

	return modin

def bern_std(bern_m, std_m, to_std=False):

	def get_bern_module(m, strin):

		_m, _name_list = m, strin.split(".")
		# update _modules with pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.add_module
		for _tmp in _name_list:
			_m = _m._modules[_tmp]
		return _m

	with torch_no_grad():
		for _name, _para in std_m.named_parameters():
			_bern_m = get_bern_module(bern_m, _name)
			if isinstance(_bern_m, BernoulliParameter):
				if to_std:
					_para.copy_(_bern_m.data)
				else:
					_bern_m.data.copy_(_para)

	return std_m if to_std else bern_m
