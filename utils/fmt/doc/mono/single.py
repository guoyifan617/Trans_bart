#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, map_batch, pad_batch
from utils.fmt.doc.base import doc_reader
#from utils.fmt.doc.mono.base import map_batch

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	nd = maxlen = minlen = mlen_i = nsent = 0
	for i_d, lgth in doc_reader(finput):
		cur_nsent = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
			nsent = cur_nsent
		if (cur_nsent == nsent) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i, nsent
			rsi = [i_d]
			mlen_i = lgth
			nsent = cur_nsent
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
			nd = 1
	if rsi:
		yield rsi, mlen_i, nsent

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, mlen_i, nsent in _batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		yield rsi, mlen_i + extok_i, nsent

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper if custom_batch_mapper is None else custom_batch_mapper
	for i_d, mlen_i, nsent in _batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield pad_batch(i_d, mlen_i), nsent
