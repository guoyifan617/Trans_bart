#encoding: utf-8

from utils.fmt.doc.mono.base import map_batch as map_batch_pret
from utils.fmt.doc.para.dual import batch_loader, batch_padder as batch_padder_base
from utils.fmt.vocab.base import map_batch as map_batch_base

map_batch = (map_batch_base, map_batch_pret,)

def batch_mapper(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, map_batch=map_batch, batch_loader=batch_loader, **kwargs):

	_map_batch_base, _map_batch_pret = map_batch
	for i_d, td, mlen_i, mlen_t, nsent in batch_loader(finput, ftarget, bsize, maxpad, maxpart, maxtoken, minbsize, **kwargs):
		rsi, extok_i = _map_batch_base(i_d, vocabi)
		rst, extok_t = _map_batch_pret(td, vocabt)
		yield rsi, rst, mlen_i + extok_i, mlen_t + extok_t, nsent

def batch_padder(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, batch_mapper=batch_mapper, **kwargs):

	return batch_padder_base(finput, ftarget, vocabi, vocabt, bsize, maxpad, maxpart, maxtoken, minbsize, batch_mapper=batch_mapper, **kwargs)
