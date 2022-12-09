#encoding: utf-8

import torch
from math import ceil
from random import shuffle
from threading import Lock
from time import sleep

from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.thread import LockHolder, start_thread, thread_keeper

from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

eos_token = "<eos>"

def sent_file_reader(fname, max_len=510, print_func=print):

	if print_func is not None:
		print_func("read %s" % fname)
	_max_len = max_len + 1
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if len(tmp) < _max_len:
					yield list(tmp)
	if print_func is not None:
		print_func("close %s" % fname)

def doc_file_reader(fname, max_len=510, eos_token=eos_token, print_func=print):

	if print_func is not None:
		print_func("read %s" % fname)
	prev_sent, prev_sent_len = None, 0
	_max_len_s = max_len + 1
	_max_len_p = _max_len_s if eos_token is None else max_len
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				_cur_l = len(tmp)
				if _cur_l < _max_len_s:
					if prev_sent is None:
						prev_sent, prev_sent_len = tmp, _cur_l
					else:
						if (prev_sent_len + _cur_l) < _max_len_p:
							_rs = list(prev_sent)
							if eos_token is None:
								_rs.append(eos_token)
							_rs.extend(list(tmp))
							yield _rs
							prev_sent, prev_sent_len = None, 0
						else:
							yield list(prev_sent)
							prev_sent, prev_sent_len = tmp, _cur_l
				else:
					if prev_sent is not None:
						yield list(prev_sent)
						prev_sent, prev_sent_len = None, 0
			else:
				if prev_sent is not None:
					yield list(prev_sent)
					prev_sent, prev_sent_len = None, 0
		if prev_sent is not None:
			yield list(prev_sent)
			prev_sent, prev_sent_len = None, 0
	if print_func is not None:
		print_func("close %s" % fname)

def file_loader(sfiles, dfiles, max_len=510, sent_file_reader=sent_file_reader, doc_file_reader=doc_file_reader, print_func=print):

	_files = [sent_file_reader(_, max_len=max_len, print_func=print_func) for _ in sfiles]
	_files.extend([doc_file_reader(_, max_len=max_len, print_func=print_func) for _ in dfiles])
	_num_line = 0
	while _files:
		_cl = []
		for i, _f in enumerate(_files):
			_data = next(_f, None)
			if _data is None:
				_cl.append(i)
			else:
				yield _data
				if print_func is not None:
					_num_line += 1
					if _num_line % 1000000 == 0:
						print_func("%d lines loaded" % _num_line)
		if _cl:
			for _ in reversed(_cl):
				del _files[_]

def inf_file_load(*args, print_func=print, **kwargs):

	_epoch = 0
	while True:
		yield from file_loader(*args, print_func=print_func, **kwargs)
		if print_func is not None:
			_epoch += 1
			print_func("Epoch: %d" % _epoch)

def sort_list_file_reader(x, *args, **kwargs):

	_d = {}
	for _ in x:
		_k = len(_)
		if _k in _d:
			_d[_k].append(_)
		else:
			_d[_k] = [_]
	for _k in sorted(_d.keys()):
		_v = _d[_k]
		shuffle(_v)
		yield from _v

class DLoader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=8, raw_cache_size=1048576, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_load, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.out = []
		self.out_lck = Lock()
		self.running = LockHolder(True)
		self.t = start_thread(target=thread_keeper, args=((self.running,), all, self.sleep_secs,), kwargs={"target": self.loader})

	def loader(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=self.print_func)
		_cpu = torch.device("cpu")
		while self.running():
			with self.out_lck:
				_num_build = self.num_cache - len(self.out)
			if _num_build > 0:
				for i in range(ceil(_num_build / 2.0)):
					_cache = []
					for _ in range(self.raw_cache_size):
						_data = next(dloader, None)
						if _data is None:
							if self.print_func is not None:
								self.print_func("end of data")
						else:
							_cache.append(_data)
					_cache = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(_cache, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=sort_list_file_reader, map_batch=map_batch, pad_id=pad_id)]
					shuffle(_cache)
					_l = len(_cache)
					if _l > 1:
						_ind = _l // 2
						with self.out_lck:
							self.out.append(_cache[:_ind])
							self.out.append(_cache[_ind:])
					else:
						with self.out_lck:
							self.out.append(_cache)
			else:
				sleep(self.sleep_secs)

	def __call__(self, *args, **kwargs):

		while self.running():
			with self.out_lck:
				if len(self.out) > 0:
					_ = self.out.pop(0)
				else:
					_ = None
			if _ is None:
				sleep(self.sleep_secs)
			else:
				yield from _
