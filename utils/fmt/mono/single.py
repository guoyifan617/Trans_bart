#encoding: utf-8

from utils.fmt.mono.base import map_batch
from utils.fmt.single import batch_padder as batch_padder_base

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs):

	return batch_padder_base(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
