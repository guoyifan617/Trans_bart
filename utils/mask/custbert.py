#encoding: utf-8

import torch

from utils.torch.comp import mask_tensor_type, torch_all_wodim, torch_any_dim
from utils.torch.ext import multinomial

from cnfg.vocab.plm.custbert import eos_id, init_token_id, mask_id, pad_id, vocab_size

def get_sptok_mask(seqin):

	_d_mask = seqin.eq(pad_id)
	_d_mask |= seqin.eq(eos_id)
	_d_mask.select(1, 0).fill_(1)

	return _d_mask

def get_mlm_mask(seqin, p=0.15):

	_n = seqin.numel()
	_m = torch.randperm(_n, dtype=torch.int32, device=seqin.device).view_as(seqin).lt(max(1, int(_n * p)))
	_sp_mask = get_sptok_mask(seqin)
	_m &= ~_sp_mask
	_m_b = torch_any_dim(_m, 1, keepdim=False)
	if not torch_all_wodim(_m_b).item():
		_m_b = ~_m_b
		_ = _sp_mask[_m_b]
		_p = _.new_full(_.size(), p, dtype=torch.float).masked_fill_(_, 0.0)
		_m[_m_b] = _m[_m_b].scatter_(1, multinomial(_p, 1, replacement=False, dim=-1), 1)

	return _m

def get_batch(seqin, p=0.15, p_mask=0.8/(1.0-0.1), p_rand=0.1, mask_id=mask_id, start_id=init_token_id, end_id=vocab_size):

	_mlm_mask = get_mlm_mask(seqin, p=p)
	_p = seqin.new_full(tuple(1 for _ in range(seqin.dim())), p_mask, dtype=torch.float).expand_as(seqin)
	_m_mask = _p.bernoulli().to(mask_tensor_type, non_blocking=True)
	_m_mask &= _mlm_mask
	rs = seqin.masked_fill(_m_mask, mask_id)
	_r_mask = _p.fill_(p_rand).bernoulli().to(mask_tensor_type, non_blocking=True)
	_r_mask &= _mlm_mask
	_n_r_mask = _r_mask.int().sum().item()
	if _n_r_mask > 0:
		rs.masked_scatter_(_r_mask, torch.randint(start_id, end_id, (_n_r_mask,), dtype=rs.dtype, device=rs.device))

	return rs, _mlm_mask
