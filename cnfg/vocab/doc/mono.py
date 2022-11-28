#encoding: utf-8

from cnfg.vocab.base import eos_id, pad_id, sos_id, use_unk

cls_id = 3
mask_id = 4

if use_unk:
	unk_id = 5
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id, "<cls>":cls_id, "<mask>":mask_id, "<unk>":unk_id}
	init_normal_token_id = 6
else:
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id, "<cls>":cls_id, "<mask>":mask_id}
	init_normal_token_id = 5
init_token_id = 5
