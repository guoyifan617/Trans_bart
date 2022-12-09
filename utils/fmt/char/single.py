#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader as file_reader
from utils.fmt.single import batch_padder as batch_padder_base

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	nd = maxlen = minlen = mlen_i = 0
	for i_d in file_reader(finput, keep_empty_line=True):
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i
			rsi = [i_d]
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=batch_loader, custom_batch_mapper=None, **kwargs):

	return batch_padder_base(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper, **kwargs)
