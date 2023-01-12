#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, iter_to_int, pad_batch

from cnfg.vocab.gector.edit import pad_id as edit_pad_id
from cnfg.vocab.gector.op import pad_id as op_pad_id
from cnfg.vocab.plm.custbert import pad_id as mlm_pad_id

pad_id = (mlm_pad_id, edit_pad_id, op_pad_id,)

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, file_reader=None, iter_to_int=iter_to_int, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	rse = []
	rst = []
	nd = maxlen = mlen = 0
	for i_d, ed, td in (finput if file_reader is None else file_reader(finput)):
		i_d, ed, td = list(iter_to_int(i_d)), list(iter_to_int(ed)), list(iter_to_int(td))
		lgth = len(i_d)
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d)
			rse.append(ed)
			rst.append(td)
			if lgth > mlen:
				mlen = lgth
			nd += 1
		else:
			yield rsi, rse, rst, mlen
			rsi = [i_d]
			rse = [ed]
			rst = [td]
			mlen = lgth
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rse, rst, mlen

def batch_padder(finput, bsize, maxpad, maxpart, maxtoken, minbsize, pad_batch=pad_batch, batch_loader=batch_loader, pad_id=pad_id, **kwargs):

	_mlm_pad_id, _edit_pad_id, _op_pad_id = pad_id
	for i_d, ed, td, mlen in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		yield pad_batch(i_d, mlen, pad_id=_mlm_pad_id), pad_batch(ed, mlen, pad_id=_edit_pad_id), pad_batch(td, mlen, pad_id=_op_pad_id)
