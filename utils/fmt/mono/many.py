#encoding: utf-8

from utils.fmt.many import batch_loader_many, batch_padder_many as batch_padder_many_base
from utils.fmt.mono.base import map_batch
from utils.fmt.mono.single import batch_padder as batch_padder_single

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

	return batch_padder_many_base(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=batch_mapper_many if custom_batch_mapper is None else custom_batch_mapper)

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
		else:
			return batch_padder_single(filelist[0], vocablist[0], bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
	else:
		return batch_padder_single(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=custom_batch_mapper)
