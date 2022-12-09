#encoding: utf-8

from math import ceil

from utils.fmt.base import get_bsize, list_reader, pad_batch
from utils.fmt.mulang.single import batch_padder as batch_padder_single
from utils.fmt.vocab.base import map_batch

def batch_loader_many(filelist, bsize, maxpad, maxpart, maxtoken, minbsize):

	_f_maxpart = float(maxpart)
	rs = [[] for i in range(len(filelist) + 1)]
	nd = maxlen = 0
	mlen = None
	for lines in zip(*[list_reader(f, keep_empty_line=True) for f in filelist]):
		lens = [len(line) for line in lines]
		lens[0] -= 1
		lgth = sum(lens)
		src_line = lines[0]
		# uncomment the following 2 lines to filter out empty data (e.g. in OPUS-100).
		#if any(_len <= 0 for _len in lens):
			#continue
		if maxlen == 0:
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			mlen = lens
		if (nd < minbsize) or (lgth <= maxlen and nd < _bsize):
			rs[0].append(src_line[1:])
			for line, rsu in zip(lines[1:], rs[1:]):
				rsu.append(line)
			rs[-1].append(src_line[0])
			for cur_len, (i, mlenu,) in zip(lens, enumerate(mlen)):
				if cur_len > mlenu:
					mlen[i] = cur_len
			nd += 1
		else:
			yield rs, mlen
			rs = [[src_line[1:]]]
			rs.extend([[line] for line in lines[1:]])
			rs.append([src_line[0]])
			mlen = lens
			maxlen = lgth + min(maxpad, ceil(lgth / _f_maxpart))
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rs:
		yield rs, mlen

def batch_mapper_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader_many if custom_batch_loader is None else custom_batch_loader
	vocabtask = vocablist[-1]
	for _rs, _mlen in _batch_loader(filelist, bsize, maxpad, maxpart, maxtoken, minbsize):
		rs = []
		mlen = []
		for rsu, mlenu, vocab in zip(_rs, _mlen, vocablist):
			_rs, extok = map_batch(rsu, vocab)
			rs.append(_rs)
			mlen.append(mlenu + extok)
		rs.append([vocabtask[tmp] for tmp in _rs[-1]])
		yield rs, mlen

def batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	_batch_mapper = batch_mapper_many if custom_batch_mapper is None else custom_batch_mapper
	for rs, mlen in _batch_mapper(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader):
		yield *tuple(pad_batch(rsu, mlenu) for rsu, mlenu in zip(rs, mlen)), rs[-1]

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
		else:
			return batch_padder_single(filelist[0], *vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
	else:
		return batch_padder_single(filelist, *vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
