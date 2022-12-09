#encoding: utf-8

from utils.fmt.many import batch_padder_many as batch_padder_many_base
from utils.fmt.mono.base import map_batch
from utils.fmt.mono.single import batch_padder as batch_padder_single

def batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs):

	return batch_padder_many_base(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)

def batch_padder(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs):

	if isinstance(filelist, (list, tuple,)):
		if len(filelist) > 1:
			return batch_padder_many(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
		else:
			return batch_padder_single(filelist[0], vocablist[0], bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
	else:
		return batch_padder_single(filelist, vocablist, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
