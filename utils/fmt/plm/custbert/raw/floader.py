#encoding: utf-8

import torch
from math import ceil
from multiprocessing import Manager, Value
from numpy import array as np_array, int32 as np_int32
from os import remove
from os.path import exists as fs_check
from random import shuffle
from threading import Lock
from time import sleep
from uuid import uuid1 as uuid_func

from utils.base import mkdir
from utils.fmt.base import seperate_list
from utils.fmt.plm.custbert.raw.base import inf_file_loader, sort_list_file_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.h5serial import h5File
from utils.process import process_keeper, start_process

from cnfg.ihyp import h5_libver, h5modelwargs, max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

cache_file_prefix = "train"

def get_cache_path(*fnames):

	_cache_path = None
	for _t in fnames:
		_ = _t.rfind("/") + 1
		if _ > 0:
			_cache_path = _t[:_]
			break
	if _cache_path is None:
		_cache_path = "cache/floader/%s/" % uuid_func().hex
	mkdir(_cache_path)

	return _cache_path

def get_cache_fname(fpath, i=0, fprefix=cache_file_prefix):

	return "%s%s.%d.h5" % (fpath, fprefix, i,)

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=8, raw_cache_size=4194304, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.cache_path = get_cache_path(*self.sent_files, *self.doc_files)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.clean_cache_file()
		self.manager = Manager()
		self.out = self.manager.list()
		self.running = Value("d", 1)
		self.p_loader = start_process(target=process_keeper, args=(self.running, self.sleep_secs,), kwargs={"target": self.loader})

	def clean_cache_file(self):

		for i in range(self.num_cache):
			_cache_file = get_cache_fname(self.cache_path, i=i, fprefix=cache_file_prefix)
			if fs_check(_cache_file):
				try:
					remove(_cache_file)
				except Exception as e:
					if self.print_func is not None:
						self.print_func(e)

	def loader(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=self.print_func)
		_cpu = torch.device("cpu")
		while self.running.value:
			for i in range(self.num_cache):
				_raw = []
				for _ in range(self.raw_cache_size):
					_data = next(dloader, None)
					if _data is None:
						if self.print_func is not None:
							self.print_func("end of file stream")
					else:
						_raw.append(_data)
				_cache_file = get_cache_fname(self.cache_path, i=i, fprefix=cache_file_prefix)
				while fs_check(_cache_file):
					sleep(self.sleep_secs)
				with h5File(_cache_file, "w", libver=h5_libver) as rsf:
					src_grp = rsf.create_group("src")
					curd = 0
					for i_d in batch_padder(_raw, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=sort_list_file_reader, map_batch=map_batch, pad_id=pad_id):
						src_grp.create_dataset(str(curd), data=np_array(i_d, dtype=np_int32), **h5modelwargs)
						curd += 1
					rsf["ndata"] = np_array([curd], dtype=np_int32)
				self.out.append(_cache_file)

	def __call__(self, *args, **kwargs):

		while self.running.value:
			if self.out:
				_fname = self.out.pop(0)
				if fs_check(_fname):
					try:
						td = h5File(_fname, "r")
					except Exception as e:
						td = None
						if self.print_func is not None:
							self.print_func(e)
					if td is not None:
						tl = [str(i) for i in range(td["ndata"][()].item())]
						shuffle(tl)
						src_grp = td["src"]
						for i_d in tl:
							yield torch.from_numpy(src_grp[i_d][()])
						td.close()
					if fs_check(_fname):
						try:
							remove(_fname)
						except Exception as e:
							if self.print_func is not None:
								self.print_func(e)
			else:
				sleep(self.sleep_secs)

	def status(self, mode=True):

		self.running.value = 1 if mode else 0

	def close(self):

		self.running.value = 0
		if self.out:
			for _fname in self.out:
				if fs_check(_fname):
					try:
						remove(_fname)
					except Exception as e:
						if self.print_func is not None:
							self.print_func(e)
			self.out.clear()
