#encoding: utf-8

from utils.fmt.dual import batch_padder as batch_padder_base
from utils.fmt.mono.base import map_batch

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, **kwargs)
