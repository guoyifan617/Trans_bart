#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader, map_batch, pad_batch
from utils.fmt.single import batch_padder as batch_padder_single

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
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
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

def batch_mapper_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader_many if custom_batch_loader is None else custom_batch_loader
	for _rs, _mlen in _batch_loader(filelist, bsize, maxpad, maxpart, maxtoken, minbsize):
		rs = []
		mlen = []
		for rsu, mlenu, vocab in zip(_rs, _mlen, vocablist):
			_rs, extok = map_batch(rsu, vocab)
			rs.append(_rs)
			mlen.append(mlenu + extok)
		yield rs, mlen

def batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper_many if custom_batch_mapper is None else custom_batch_mapper
	for rs, mlen in _batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield tuple(pad_batch(rsu, mlenu) for rsu, mlenu in zip(rs, mlen))

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
		else:
			return batch_padder_single(filelist[0], vocablist[0], bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
	else:
		return batch_padder_single(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)