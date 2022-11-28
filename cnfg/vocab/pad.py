#encoding: utf-8

from cnfg.hyp import use_unk

pad_id = 0
if use_unk:
	unk_id = 1
	init_vocab = {"<pad>":pad_id}
	init_normal_token_id = 2
else:
	unk_id = None
	init_vocab = {"<pad>":pad_id}
	init_normal_token_id = 1
init_token_id = 2
