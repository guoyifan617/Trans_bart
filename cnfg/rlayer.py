#encoding: utf-8

from cnfg.base import *

mana = "enc"

layer_linear = 0

run_id += ".%s.%d" % (mana, layer_linear,)

earlystop = maxrun

def add_out_func_dec(*inputs, **kwargs):

	return kwargs["query_unit"] if "query_unit" in kwargs else inputs[-1]

def add_out_func_rfn_enc(*inputs, **kwargs):

	return inputs[1]

def add_out_func_rfn_dec_train(*inputs, **kwargs):

	return inputs[2]

def add_out_func_rfn_dec(*inputs, **kwargs):

	return kwargs["query_unit"] if "query_unit" in kwargs else inputs[-1], inputs[2]

if mana.lower() == "enc":
	model_func = lambda m: m.enc
	arg_index = 0
	kwargs_key = None
	add_out_func = None#add_out_func_rfn_enc
else:
	model_func = lambda m: m.dec
	arg_index = 1#-1 for decoding
	kwargs_key = None#"query_unit"
	add_out_func = add_out_func_dec#add_out_func_rfn_dec_train
