#encoding: utf-8

# pad_id, keep_id, delete_id must be consistent with cnfg.vocab.gector.op for utils.fmt.gec.gector.apply_op to work correctly
pad_id, keep_id, delete_id, mask_id, insert_id, blank_id = 0, 1, 2, 3, 4, 5
init_vocab = {"<pad>":pad_id, "k":keep_id, "d":delete_id, "m":mask_id, "i":insert_id, "b":blank_id}
vocab_size = 6
