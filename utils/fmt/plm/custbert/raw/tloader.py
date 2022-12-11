#encoding: utf-8

import torch
from math import ceil
from random import shuffle
from threading import Lock
from time import sleep

from utils.fmt.base import seperate_list
from utils.fmt.plm.custbert.raw.base import inf_file_loader, sort_list_file_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.thread import LockHolder, start_thread, thread_keeper

from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=512, raw_cache_size=1048576, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
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
								self.print_func("end of file stream")
						else:
							_cache.append(_data)
					_cache = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(_cache, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=sort_list_file_reader, map_batch=map_batch, pad_id=pad_id)]
					shuffle(_cache)
					_cache = seperate_list(_cache, self.nbatch)
					with self.out_lck:
						self.out.extend(_cache)
			else:
				sleep(self.sleep_secs)

	def __call__(self, *args, **kwargs):

		while self.running():
			with self.out_lck:
				if len(self.out) > 0:
					_ = self.out.pop(0)
				else:
					_ = None
			if _ is not None:
				yield from _

	def status(self, mode=True):

		self.running(mode)

	def close(self):

		self.running(False)
		with self.out_lck:
			self.out.clear()
