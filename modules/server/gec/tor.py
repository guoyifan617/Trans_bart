#encoding: utf-8

import torch
from math import ceil

from parallel.parallelMT import DataParallelMT
from transformer.GECToR.NMT import NMT
from utils.fmt.base import dict_insert_set, get_bsize, iter_dict_sort
from utils.fmt.base4torch import parse_cuda_decode
from utils.fmt.plm.custbert.single import batch_padder
from utils.fmt.plm.custbert.token import Tokenizer
from utils.fmt.vocab.base import reverse_dict
from utils.io import load_model_cpu
from utils.torch.comp import torch_autocast, torch_compile, torch_inference_mode

from cnfg.ihyp import *
from cnfg.vocab.plm.custbert import eos_id, sos_id, vocab_size

def load_fixing(module):
	if hasattr(module, "fix_load"):
		module.fix_load()

def batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize, get_bsize=get_bsize, **kwargs):

	_f_maxpart = float(maxpart)
	rsi = []
	nd = maxlen = minlen = mlen_i = 0
	for i_d in finput:
		i_d = list(i_d)
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i
			rsi = [i_d]
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, ceil(lgth / _f_maxpart)) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def sorti(lin):

	data = {}
	for ls in lin:
		data = dict_insert_set(data, ls, len(ls))
	for _ in iter_dict_sort(data, free=True):
		yield from _

class Handler:

	def __init__(self, modelfs, cnfg, minbsize=1, expand_for_mulgpu=True, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, norm_u8=False, **kwargs):

		self.tokenizer = Tokenizer(cnfg.plm_vcb, norm_u8=norm_u8, post_norm_func=None, split=False)
		self.vcbt = reverse_dict(self.tokenizer.vcb)

		if expand_for_mulgpu:
			self.bsize = bsize * minbsize
			self.maxtoken = maxtoken * minbsize
		else:
			self.bsize = bsize
			self.maxtoken = maxtoken
		self.maxpad = maxpad
		self.maxpart = maxpart
		self.minbsize = minbsize

		model = NMT(cnfg.isize, vocab_size, vocab_size, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.act_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)
		model.build_task_model(fix_init=False)
		model = load_model_cpu(modelfs, model)
		model.apply(load_fixing)
		model.eval()
		self.use_cuda, self.cuda_device, cuda_devices, self.multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)
		if self.use_cuda:
			model.to(self.cuda_device, non_blocking=True)
			if self.multi_gpu:
				model = DataParallelMT(model, device_ids=cuda_devices, output_device=self.cuda_device.index, host_replicate=True, gather_output=False)
		self.net = torch_compile(model, *torch_compile_args, **torch_compile_kwargs)
		self.use_amp = cnfg.use_amp and self.use_cuda
		self.beam_size = cnfg.beam_size
		self.length_penalty = cnfg.length_penalty

	def __call__(self, sentences_iter, **kwargs):

		_tok_ids = [tuple(self.tokenizer(_)) for _ in sentences_iter]
		_sorted_token_ids = list(sorti(_tok_ids))
		_vcbt, _cuda_device, _multi_gpu, _use_amp = self.vcbt, self.cuda_device, self.multi_gpu, self.use_amp
		rs = []
		with torch_inference_mode():
			for seq_batch in batch_padder(_sorted_token_ids, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, batch_loader=batch_loader):
				_seq_batch = torch.as_tensor(seq_batch, dtype=torch.long, device=_cuda_device)
				with torch_autocast(enabled=_use_amp):
					output = self.net.decode(_seq_batch, beam_size=self.beam_size, max_len=None, length_penalty=self.length_penalty)
				if _multi_gpu:
					tmp = []
					for ou in output:
						tmp.extend(ou)
					output = tmp
				for tran in output:
					tmp = []
					_ = tran.tolist()
					if _[0] == sos_id:
						_ = _[1:]
					for tmpu in _:
						if tmpu == eos_id:
							break
						else:
							tmp.append(_vcbt[tmpu])
					rs.append("".join(tmp))
				_seq_batch = None
		_mapd = {_k: _v for _k, _v in zip(_sorted_token_ids, rs)}

		return [_mapd.get(_, "") for _ in _tok_ids]
