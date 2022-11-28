#encoding: utf-8

from utils.fmt.base import map_batch
from utils.fmt.doc.mono.base import map_batch as map_batch_pret
from utils.fmt.doc.para.dual import batch_loader, batch_padder as batch_padder_base

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None):

	_batch_loader = batch_loader if custom_batch_loader is None else custom_batch_loader
	for i_d, td, mlen_i, mlen_t, nsent in _batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi, extok_i = map_batch(i_d, vocabi)
		rst, extok_t = map_batch_pret(td, vocabt)
		yield rsi, rst, mlen_i + extok_i, mlen_t + extok_t, nsent

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=None, custom_batch_mapper=None):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, custom_batch_loader=custom_batch_loader, custom_batch_mapper=batch_mapper if custom_batch_mapper is None else custom_batch_mapper)
