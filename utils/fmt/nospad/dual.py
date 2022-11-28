#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader
from utils.fmt.dual import batch_padder as batch_padder_base

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(list_reader(finput, keep_empty_line=True), list_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d)
		ltd = len(td)
		lgth = lid + ltd
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			mlen_i = lid
		if (lid == mlen_i) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			rsi.append(i_d)
			rst.append(td)
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, mlen_i, mlen_t
			rsi = [i_d]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, mlen_i, mlen_t

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=batch_loader if custom_batch_loader is None else custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
