#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, line_reader, list_reader, pad_batch
from utils.fmt.vocab.base import map_batch

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist))]
	nd = maxlen = 0
	mlen = None
	for lines in zip(*([list_reader(f, keep_empty_line=True) for f in filelist[:-1]] + [line_reader(filelist[-1], keep_empty_line=True)])):
		lens = [len(line) for line in lines[:-1]]
		lgth = sum(lens)
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			mlen = lens
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			for line, rsu in zip(lines[:-1], rs):
				rsu.append(line)
			rs[-1].append(float(lines[-1]))
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, mlen
			rs = [[line] for line in lines[:-1]]
			rs.append([float(lines[-1])])
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
		rs.append(_rs[-1])
		yield rs, mlen

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper_many if custom_batch_mapper is None else custom_batch_mapper
	for rs, mlen in _batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield *tuple(pad_batch(rsu, mlenu) for rsu, mlenu in zip(rs, mlen)), rs[-1]
