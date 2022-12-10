#encoding: utf-8

import torch
from math import ceil
from multiprocessing import Queue, Value
from random import shuffle

from utils.fmt.plm.custbert.raw.base import inf_file_loader, sort_list_file_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.process import process_keeper, start_process

from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=4096, raw_cache_size=1048576, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.out = Queue()
		self.running = Value("d", 1)
		self.t = start_process(target=process_keeper, args=(self.running, self.sleep_secs,), kwargs={"target": self.loader})

	def loader(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=self.print_func)
		_cpu = torch.device("cpu")
		_cache = []
		while self.running.value:
			_num_cache = len(_cache)
			if _num_cache < self.num_cache:
				_raw = []
				for _ in range(self.raw_cache_size):
					_data = next(dloader, None)
					if _data is None:
						if self.print_func is not None:
							self.print_func("end of data")
					else:
						_raw.append(_data)
				_raw = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(_raw, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=sort_list_file_reader, map_batch=map_batch, pad_id=pad_id)]
				shuffle(_raw)
				_cache.extend(_raw)
				_raw = None
			_num_put = self.num_cache - self.out.qsize()
			if _num_put > 0:
				_num_cache = len(_cache)
				if _num_cache > 0:
					_nput = min(_num_put, _num_cache)
					for _ in range(_nput):
						self.out.put(_cache.pop(0))

	def __call__(self, *args, **kwargs):

		while self.running.value:
			for _ in range(self.out.qsize()):
				if self.out.empty():
					break
				else:
					yield self.out.get()

	def status(self, mode=True):

		self.running.value = 1 if mode else 0

	def close(self):

		self.running.value = 0
		while not self.out.empty():
			self.out.get()