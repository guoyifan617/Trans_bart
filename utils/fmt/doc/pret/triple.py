#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, map_batch, pad_batch
from utils.fmt.doc.base import doc_reader
from utils.fmt.doc.mono.base import map_batch as map_batch_pret

def batch_loader(finput, fpret, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rsp = []
	rst = []
	nd = maxlen = minlen = mlen_i = mlen_p = mlen_t = nsent = 0
	for (i_d, i_lgth), (pd, p_lgth), (td, t_lgth) in zip(doc_reader(finput), doc_reader(fpret), doc_reader(ftarget)):
		cur_nsent = len(i_d)
		lgth = i_lgth + p_lgth + t_lgth
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
			nsent = cur_nsent
		if (cur_nsent == nsent) and ((nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize)):
			rsi.append(i_d)
			rsp.append(pd)
			rst.append(td)
			if i_lgth > mlen_i:
				mlen_i = i_lgth
			if p_lgth > mlen_p:
				mlen_p = p_lgth
			if t_lgth > mlen_t:
				mlen_t = t_lgth
			nd += 1
		else:
			yield rsi, rsp, rst, mlen_i, mlen_p, mlen_t, nsent
			rsi = [i_d]
			rsp = [pd]
			rst = [td]
			mlen_i = i_lgth
			mlen_p = p_lgth
			mlen_t = t_lgth
			nsent = cur_nsent
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = max(1, get_bsize(maxlen, maxtoken, bsize) // cur_nsent)
			nd = 1
	if rsi:
		yield rsi, rsp, rst, mlen_i, mlen_p, mlen_t, nsent

def batch_mapper(finput, fpret, ftarget, vocabi, vocabp, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, pd, td, mlen_i, mlen_p, mlen_t, nsent in _batch_loader(finput, fpret, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rsp, extok_p = map_batch_pret(pd, vocabp)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rsp, rst, mlen_i + extok_i, mlen_p + extok_p, mlen_t + extok_t, nsent

def batch_padder(finput, fpret, ftarget, vocabi, vocabp, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper if custom_batch_mapper is None else custom_batch_mapper
	for i_d, pd, td, mlen_i, mlen_p, mlen_t, nsent in _batch_mapper(finput, fpret, ftarget, vocabi, vocabp, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield pad_batch(i_d, mlen_i), pad_batch(pd, mlen_p), pad_batch(td, mlen_t), nsent
