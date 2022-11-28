#encoding: utf-8

import torch

# below values shall be consistent with data pre-processing

from cnfg.vocab.base import eos_id, init_normal_token_id, pad_id

def build_rand_batch(batchin, vcbsize):

	bsize, seql = batchin.size()
	construct_len = seql - 1
	rinds = torch.randint(init_normal_token_id, vcbsize, (bsize, construct_len,), dtype=batchin.dtype, device=batchin.device)
	_batch_tail = batchin.narrow(1, 1, construct_len)
	pad_mask, eos_mask = _batch_tail.eq(pad_id), _batch_tail.eq(eos_id)
	rinds.masked_fill_(pad_mask, pad_id).masked_fill_(eos_mask, eos_id)

	return torch.cat((batchin.narrow(1, 0, 1), rinds,), dim=1), pad_mask
