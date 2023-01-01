#encoding: utf-8

from random import randint

from utils.mask.base import get_sind, mask_rand_token, mask_token

from cnfg.vocab.mono import pad_id

def get_batch(batch_in, p_ext, p_mask, p_rand, mask_id, start_id, end_id, pad_id=pad_id):

	bsize, nsent, seql = batch_in.size()
	sel_sent = randint(0, nsent - 1)
	batch_sel = batch_in.select(1, sel_sent)
	seql = seql - 1 - batch_sel.eq(pad_id).sum(-1).min().item()
	tgt_batch = batch_sel.narrow(1, 1, seql).clone()
	_sind, _elen = get_sind(seql, p_ext, max(0, seql - 2 - tgt_batch.eq(pad_id).sum(-1).max().item()))
	mask_rand_token(mask_token(batch_sel.narrow(1, _sind + 2, _elen - 1), p_mask, mask_id), p_rand, start_id, end_id)
	tgt_batch = tgt_batch.narrow(1, _sind, _elen)

	return batch_in, tgt_batch, sel_sent, _sind
