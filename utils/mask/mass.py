#encoding: utf-8

from utils.mask.base import get_sind, mask_rand_token, mask_token

from cnfg.vocab.mono import pad_id

def get_batch(batch_in, p_ext, p_mask, p_rand, mask_id, start_id, end_id, pad_id=pad_id):

	seql = batch_in.size(-1)
	_sind, _elen = get_sind(seql, p_ext, max(0, seql - 2 - batch_in.eq(pad_id).sum(-1).max().item()))
	tgt_batch = batch_in.narrow(1, _sind, _elen).clone()
	sel_batch = batch_in.narrow(1, _sind + 1, _elen - 1)
	mask_rand_token(mask_token(sel_batch, p_mask, mask_id), p_rand, start_id, end_id)

	return batch_in, tgt_batch, _sind
