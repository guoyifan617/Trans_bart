#encoding: utf-8

from utils.fmt.base import list_reader, no_unk_mapper

from cnfg.vocab.mono import *

def ldvocab(vfile, minf=False, omit_vsize=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id):

	rs = init_vocab.copy()
	cwd = init_normal_token_id
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	for data in list_reader(vfile, keep_empty_line=False):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					for wd in data[1:]:
						rs[wd] = cwd
						cwd += 1
				else:
					for wd in data[1:vsize + 1]:
						rs[wd] = cwd
						cwd += 1
						ndata = vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				for wd in data[1:]:
					rs[wd] = cwd
					cwd += 1
		else:
			break
	return rs, cwd

def map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch_core(idu, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs) for idu in i_d]
	else:
		rsi = [sos_id]
		rsi.extend([vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d))#[vocabi[wd] for wd in i_d if wd in vocabi]
		rsi.append(eos_id)
		return rsi

def map_batch(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs):

	return map_batch_core(i_d, vocabi, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id, **kwargs), 2
