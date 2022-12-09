#encoding: utf-8

from utils.fmt.vocab.base import no_unk_mapper

from cnfg.hyp import use_unk
from cnfg.vocab.base import unk_id

def map_batch_core(i_d, vocabi, unk_id=unk_id, **kwargs):

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch_core(idu, vocabi, unk_id=unk_id, **kwargs) for idu in i_d]
	else:
		rs = [vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d)#[vocabi[wd] for wd in i_d if wd in vocabi]
		return rs

def map_batch(i_d, vocabi, unk_id=unk_id, **kwargs):

	return map_batch_core(i_d, vocabi, unk_id=unk_id, **kwargs), 0
