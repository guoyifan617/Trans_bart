#encoding: utf-8

from cnfg.base import *

group_id = "std"

run_id = "base"

data_id = "rs"

exp_dir = "expm/"

train_data = "/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/meng2zh_bart/rs3_reverse/train.h5"
dev_data = "/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_Cache/meng2zh_bart/rs3_reverse/dev.h5"
test_data ="/home/yfguo/CCMT_model/bart_wright/transformer-edge/cache/16k_meng2zh_true_reverse_dict_trans/test.h5"

# new configurations for BART
model_name = ("encoder", "decoder",)
num_type = None
remove_classifier_bias = False
pre_trained_m = "/home/yfguo/CCMT_model/bart_wright/bart-base-chinese/pytorch_model.bin"

# override standard configurations
bindDecoderEmb = True
share_emb = False

isize = 768
ff_hsize = isize * 4
nhead = max(1, isize // 64)
attn_hsize = isize

#nlayer = [12,12]
nlayer = 6
drop = 0.1
attn_drop = drop

norm_output = True
batch_report = 200
report_eva = True

# use_cuda = True
beam_size = 5



use_cuda = False