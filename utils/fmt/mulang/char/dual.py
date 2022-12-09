#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader as file_reader
from utils.fmt.mulang.dual import batch_padder as batch_padder_base

def batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rsi = []
	rst = []
	rstask = []
	nd = maxlen = mlen_i = mlen_t = 0
	for i_d, td in zip(file_reader(finput, keep_empty_line=True), file_reader(ftarget, keep_empty_line=True)):
		_ind = i_d.find(" ")
		lid = len(i_d) - _ind - 1
		ltd = len(td)
		lgth = lid + ltd
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		#if (lid <= 0) or (ltd <= 0):
			#continue
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rsi.append(i_d[_ind + 1:])
			rstask.append(i_d[:_ind])
			rst.append(td)
			if lid > mlen_i:
				mlen_i = lid
			if ltd > mlen_t:
				mlen_t = ltd
			nd += 1
		else:
			yield rsi, rst, rstask, mlen_i, mlen_t
			rsi = [i_d[_ind + 1:]]
			rstask = [i_d[:_ind]]
			rst = [td]
			mlen_i = lid
			mlen_t = ltd
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, rst, rstask, mlen_i, mlen_t

def batch_padder(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=batch_loader, custom_batch_mapper=None, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, vocabtask, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper, **kwargs)
