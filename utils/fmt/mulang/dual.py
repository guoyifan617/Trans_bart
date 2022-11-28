#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader, map_batch, pad_batch

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	rstask = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(list_reader(finput, keep_empty_line=True), list_reader(ftarget, keep_empty_line=True)):
		lid = len(i_d) - 1
		ltd = len(td)
		lgth = lid + ltd
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		#if (lid <= 0) or (ltd <= 0):
			#continue
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d[1:])
			rstask.append(i_d[0])
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, rstask, mlen_i, mlen_t
			rsi = [i_d[1:]]
			rstask = [i_d[0]]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, rstask, mlen_i, mlen_t

def batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, td, taskd, mlen_i, mlen_t in _batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch(td, vocabt)
		yield rsi, rst, [vocabtask[tmp] for tmp in taskd], mlen_i + extok_i, mlen_t + extok_t

def batch_padder(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper if custom_batch_mapper is None else custom_batch_mapper
	for i_d, td, taskd, mlen_i, mlen_t in _batch_mapper(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield pad_batch(i_d, mlen_i), pad_batch(td, mlen_t), taskd
