#encoding: utf-8

from utils.fmt.consdec import map_batch as map_batch_no_st
from utils.fmt.dual import batch_loader, batch_padder as batch_padder_base
from utils.fmt.vocab.base import map_batch

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, td, mlen_i, mlen_t in _batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch_no_st(td, vocabt)
		yield rsi, rst, mlen_i + extok_i, mlen_t + extok_t

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=batch_mapper if custom_batch_mapper is None else custom_batch_mapper)
