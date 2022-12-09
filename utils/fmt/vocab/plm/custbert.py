#encoding: utf-8

from utils.fmt.vocab.base import map_batch as map_batch_base

from cnfg.vocab.plm.custbert import eos_id, sos_id, unk_id, use_unk

def map_batch(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	return map_batch_base(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs)
