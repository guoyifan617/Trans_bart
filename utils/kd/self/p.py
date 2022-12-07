#encoding: utf-8

import torch
from torch.nn import Module
from torch.nn.functional import kl_div

from utils.kd.base import renorm
from utils.torch.comp import exist_any, torch_any_dim

from cnfg.vocab.base import pad_id

def correct_index(i, gold):

	_g_m = i.eq(gold.unsqueeze(-1))
	_fix_m = ~torch_any_dim(_g_m, -1)
	if exist_any(_fix_m):
		i.select(-1, -1).masked_scatter_(_fix_m, gold[_fix_m])
		_g_m.select(-1, -1).masked_fill_(_fix_m, 1)

	return i, _g_m

def fix_gold_p(p, gold_ind_mask, min_gold_p):

	_p_e = p.masked_fill(gold_ind_mask, 0.0).sum(-1, keepdim=True)
	_up_es = 1.0 - min_gold_p
	_p_gt_m = _p_e.gt(_up_es)
	if exist_any(_p_gt_m):
		_se_mask = _p_gt_m & (~gold_ind_mask)
		p[_se_mask] = p[_se_mask].div_(_p_e.expand_as(p)[_se_mask]).mul_(_up_es)
		p.masked_fill_(_p_gt_m & gold_ind_mask, min_gold_p)

	return p

def fix_gold_thres(p, gold_ind_mask, min_gold_p):

	_p_max, _i_max = p.max(-1, keepdim=False)
	_e_m = _i_max.ne(gold_ind_mask.nonzero().select(-1, -1).view_as(_i_max))
	if exist_any(_e_m):
		p[gold_ind_mask & _e_m.unsqueeze(-1)] = _p_max[_e_m] + min_gold_p
		#_e_m = _e_m.expand_as(p)
		_s_p = p[_e_m]#.view(-1, p.size(-1))
		p[_e_m] = _s_p.div_(_s_p.sum(-1, keepdim=True))#.view(-1)

	return p

fix_gold = fix_gold_thres

def mwb_linear(x, weight, bias):

	_osize, _isize = weight.size()[-2:]
	_w_t = weight.view(-1, _osize, _isize).transpose(1, 2)
	_x_sizes = x.size()
	_x_v = x.view(-1, *_x_sizes[-2:])
	out = _x_v.bmm(_w_t) if bias is None else bias.view(-1, 1, _osize).baddbmm(_x_v, _w_t)

	return out.view(*_x_sizes[:-1], _osize)

def consis_ens(x):

	return x.transpose(-1, -2).matmul(x.matmul(x.mean(-2, keepdim=True).transpose(-1, -2)).softmax(-2)).squeeze(-1)

def qsel(x, gold_ind_mask):

	_s = x.size()

	return x.gather(-2, x.masked_select(gold_ind_mask.unsqueeze(-2)).view(*_s[:-1], 1).argmax(-2, keepdim=True).expand(*_s[:-2], 1, _s[-1])).squeeze(-2)

# equivalent to but slower than qsel
def index_qsel(x, gold_ind_mask):

	_s = x.size()
	_nlayer, _ntopk = _s[-2:]
	_sel_ind = x.masked_select(gold_ind_mask.unsqueeze(-2)).view(*_s[:-1]).argmax(-1).view(-1)
	_bsize = _sel_ind.size(0)

	return x.view(-1, _ntopk).index_select(0, _sel_ind + torch.arange(0, _bsize * _nlayer, _nlayer, dtype=_sel_ind.dtype, device=_sel_ind.device)).view(*_s[:-2], _ntopk)

def qens(x, gold_ind_mask):

	return x.transpose(-1, -2).matmul(x.masked_select(gold_ind_mask.unsqueeze(-2)).view(*x.size()[:-1]).log().softmax(-1).unsqueeze(-1)).squeeze(-1)

def q_consis_ens(x, gold_ind_mask):

	_xt = x.transpose(-1, -2).contiguous()

	return _xt.matmul(x.matmul(_xt.matmul(x.masked_select(gold_ind_mask.unsqueeze(-2)).view(*x.size()[:-1]).log().softmax(-1).unsqueeze(-1))).softmax(-2)).squeeze(-1)

def q_consis_p_ens(x, gold_ind_mask, p):

	return x.transpose(-1, -2).matmul(x.masked_select(gold_ind_mask.unsqueeze(-2)).view(*x.size()[:-1]).log().softmax(-1).unsqueeze(-1).mul_(p).add_(x.matmul(x.mean(-2, keepdim=True).transpose(-1, -2)).softmax(-2), alpha=1.0 - p)).squeeze(-1)

def max_renorm(x, dim):

	_ = x.amax(dim)

	return _ / _.sum(-1, keepdim=True)

def get_kd_loss(classifier=None, final_predict=None, enc_kd_o=None, kd_o=None, gold=None, gold_pad_mask=None, num_topk=None, T=1.0, min_gold_p=0.0, deep_kd=True, mix_p=None, remove_gold=False):

	# (bsize, nquery, ntopk)
	_s, _i = final_predict.detach().topk((num_topk + 1) if remove_gold else num_topk, dim=-1)
	_i, _gold_ind_mask = correct_index(_i, gold)
	# efficient implementation when quality ensemble and min_gold_p is not enabled
	#if remove_gold:
		#_ = ~_gold_ind_mask
		#_i = _i[_].view(*_i.size()[:-1], num_topk)
		# for last layer kd
		#_s = _s[_].view(*_s.size()[:-1], num_topk)

	_f_i = _i.view(-1)
	# (bsize, nquery, ntopk, isize)
	_size = _i.size()
	_c_w = classifier.weight.index_select(0, _f_i).view(*_size, -1)
	# (bsize, nquery, ntopk)
	_c_b = None if classifier.bias is None else classifier.bias.index_select(0, _f_i).view(_size)
	# (bsize, nquery, nlayer, isize)
	_kd_o = torch.stack(kd_o, dim=-2)
	if enc_kd_o is not None:
		_kd_o = torch.cat([enc_kd_o, _kd_o], dim=-2)

	# (bsize, nquery, nlayer, ntopk)
	_kd_s = mwb_linear(_kd_o, _c_w, _c_b)
	if deep_kd:
		_kd_s = torch.cat([_kd_s, final_predict.gather(-1, _i).unsqueeze(-2)], dim=-2)

	if isinstance(T, Module):
		_has_T = _learn_T = True
		_T = T()
	else:
		_has_T = (T != 1.0)
		_learn_T = False
		_T = T
	if _has_T:
		#if _learn_T:
			#_kd_t = _kd_s.detach().div(_T)
		_kd_s.div_(_T)
		#else:
			#_kd_s.div_(_T)
			#_kd_t = _kd_s.detach()

	_fix_p = (min_gold_p > 0.0)
	# last layer
	_p = _s.div_(_T).softmax(dim=-1) if _has_T else _s.softmax(dim=-1)
	if _learn_T and _fix_p:
		_p = _p.clone()
	# complex rules
	#if deep_kd:
		#_p = _kd_t.softmax(dim=-1)
	#else:
		#_ = final_predict.detach().gather(-1, _i).unsqueeze(-2)
		#_p = torch.cat([_kd_t, _.div_(T) if _has_T else _], dim=-2).softmax(dim=-1)
	# ensemble layers' outputs
	#_p = _p.mean(-2)
	# consistency ensemble
	#_p = consis_ens(_p)
	# quality selection
	#_p = qsel(_p, _gold_ind_mask)
	# quality ensemble
	#_p = qens(_p, _gold_ind_mask)
	# quality->consistency ensemble
	#_p = q_consis_ens(_p, _gold_ind_mask)
	# quality consistency parallel ensemble
	#_p = q_consis_p_ens(_p, _gold_ind_mask, 0.7)
	# max pooling, renorm is not necessary when remove_gold is True
	#_p = renorm(_p.amax(-2), dim=-1)

	if _fix_p:
		_p = fix_gold(_p, _gold_ind_mask, min_gold_p)

	# for quality ensemble
	if remove_gold:
		_ = ~_gold_ind_mask
		_kd_s = _kd_s[_.unsqueeze(-2).expand_as(_kd_s)].view(*_kd_s.size()[:-1], num_topk)
		_p = renorm(_p[_].view(*_p.size()[:-1], num_topk), dim=-1)

	_gold_pad_mask = gold.eq(pad_id) if gold_pad_mask is None else gold_pad_mask
	_p.masked_fill_(_gold_pad_mask.unsqueeze(-1), 0.0)

	if mix_p is None:
		return kl_div(_kd_s.log_softmax(-1), _p.unsqueeze(-2).expand_as(_kd_s), reduction="sum")
	else:
		return kl_div(_kd_s.transpose(-1, -2).contiguous().view(-1, mix_p.size(0)).mv(mix_p).view_as(_p).log_softmax(-1), _p, reduction="sum")

def get_iter_kd_loss(classifier=None, final_predict=None, enc_kd_o=None, kd_o=None, gold=None, gold_pad_mask=None, num_topk=None, T=1.0, min_gold_p=0.0, deep_kd=True, mix_p=None, remove_gold=False):

	_s, _i = final_predict.detach().topk((num_topk + 1) if remove_gold else num_topk, dim=-1)
	_i, _gold_ind_mask = correct_index(_i, gold)
	# efficient implementation when quality ensemble and min_gold_p is not enabled
	if remove_gold and (min_gold_p <= 0.0):
		_ = ~_gold_ind_mask
		_i = _i[_].view(*_i.size()[:-1], num_topk)

	_f_i = _i.view(-1)
	# (bsize, nquery, ntopk, isize)
	_size = _i.size()
	_c_w = classifier.weight.index_select(0, _f_i).view(*_size, -1)
	# (bsize, nquery, ntopk)
	_c_b = None if classifier.bias is None else classifier.bias.index_select(0, _f_i).view(_size)
	# (bsize, nquery, nlayer, isize)
	_kd_o = torch.stack(kd_o, dim=-2)

	if enc_kd_o is None:
		_num_enc_kd = 0
	else:
		_num_enc_kd = enc_kd_o.size(-2)
		_kd_o = torch.cat([enc_kd_o, _kd_o], dim=-2)

	# (bsize, nquery, nlayer, ntopk)
	_kd_s = mwb_linear(_kd_o, _c_w, _c_b)
	if deep_kd:
		_kd_s = torch.cat([_kd_s, final_predict.gather(-1, _i).unsqueeze(-2)], dim=-2)

	if isinstance(T, Module):
		_has_T = _learn_T = True
		_T = T()
	else:
		_has_T = (T != 1.0)
		_learn_T = False
		_T = T

	_nlayer = _kd_s.size(-2)
	_enc_kd = (_num_enc_kd > 0)
	if _enc_kd:
		_num_dec_kd = _nlayer - _num_enc_kd
		_num_o_enc_kd = _num_enc_kd - 1
		_num_o_dec_kd = _num_dec_kd - 1
		_p = torch.cat((_kd_s.narrow(-2, 1, _num_o_enc_kd), _kd_s.narrow(-2, _num_enc_kd + 1, _num_o_dec_kd),), dim=-2).detach()
		_kd_s = torch.cat((_kd_s.narrow(-2, 0, _num_o_enc_kd), _kd_s.narrow(-2, _num_enc_kd, _num_o_dec_kd),), dim=-2)
		_num_layer_kd = _num_o_enc_kd + _num_o_dec_kd
	else:
		_num_layer_kd = _nlayer - 1
		_p = _kd_s.narrow(-2, 1, _num_layer_kd).detach()
		_kd_s = _kd_s.narrow(-2, 0, _num_layer_kd)

	if _has_T:
		if _enc_kd:
			_p.div_(_T)
		else:
			_p = _p.div(_T)
		_kd_s.div_(_T)

	_p = _p.softmax(dim=-1)
	_kd_s = _kd_s.log_softmax(dim=-1)

	if min_gold_p > 0.0:
		_size = [-1 for _ in range(_gold_ind_mask.dim() + 1)]
		_size[-2] = _num_layer_kd
		_p = fix_gold(_p.clone() if _learn_T else _p, _gold_ind_mask.unsqueeze(-2).expand(_size), min_gold_p)

	_gold_pad_mask = gold.eq(pad_id) if gold_pad_mask is None else gold_pad_mask
	_p.masked_fill_(_gold_pad_mask.unsqueeze(-1).unsqueeze(-1), 0.0)

	if mix_p is None:
		return kl_div(_kd_s, _p, reduction="sum")
	else:
		return kl_div(_kd_s, _p, reduction="none").transpose(-1, -2).contiguous().view(-1, mix_p.size(0)).sum(0).dot(mix_p)
