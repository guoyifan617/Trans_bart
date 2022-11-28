#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader
from utils.fmt.many import batch_padder_many as batch_padder_many_base

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist))]
	nd = maxlen = 0
	mlen = None
	for lines in zip(*[list_reader(f, keep_empty_line=True) for f in filelist]):
		lens = [len(line) for line in lines]
		lgth = sum(lens)
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			mlen = lens
		if all(_lu == _mlenu for _lu, _mlenu in zip(lens[:-1], mlen)) and ((nd < minbsize) or (lgth <= maxlen and nd < _bsize)):
			for line, rsu in zip(lines, rs):
				rsu.append(line)
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, mlen
			rs = [[line] for line in lines]
			mlen = lens
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rs:
		yield rs, mlen

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	return batch_padder_many_base(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=batch_loader_many if custom_batch_loader is None else custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
