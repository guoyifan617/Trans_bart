#encoding: utf-8

from cnfg.hyp import use_unk

pad_id, sos_id, eos_id, mask_id  = 0, 101, 102, 103
if use_unk:
	unk_id = 100
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id, "<unk>":unk_id, "<mask>":mask_id}
	init_normal_token_id = 105
else:
	unk_id = None
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id,"<mask>":mask_id}
	init_normal_token_id = 105
init_token_id = 105
