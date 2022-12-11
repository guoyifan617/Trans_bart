#encoding: utf-8

import torch
try:
	from torch.multiprocessing import Queue, Value
except:
	from multiprocessing import Queue, Value
from math import ceil
from random import shuffle
from threading import Lock

from utils.fmt.base import seperate_list
from utils.fmt.plm.custbert.raw.base import inf_file_loader, sort_list_file_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.process import process_keeper, start_process
from utils.thread import start_thread, thread_keeper

from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=512, raw_cache_size=1048576, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.out = Queue()
		self.running = Value("d", 1)
		self.p_loader = start_process(target=process_keeper, args=(self.running, self.sleep_secs,), kwargs={"target": self.loader})
		self.t_builder = self.t_sender = None

	def builder(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=self.print_func)
		_cpu = torch.device("cpu")
		while self.running.value:
			with self.cache_lck:
				_num_cache = len(self.cache)
			if _num_cache < self.num_cache:
				_raw = []
				for _ in range(self.raw_cache_size):
					_data = next(dloader, None)
					if _data is None:
						if self.print_func is not None:
							self.print_func("end of file stream")
					else:
						_raw.append(_data)
				# as the reference to the tensor will be released after put into the queue, we cannot move it to the shared memory with .share_memory_()
				_raw = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(_raw, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=sort_list_file_reader, map_batch=map_batch, pad_id=pad_id)]
				shuffle(_raw)
				_raw = seperate_list(_raw, self.nbatch)
				with self.cache_lck:
					self.cache.extend(_raw)
				_raw = None

	def sender(self):

		while self.running.value:
			_num_put = self.num_cache - self.out.qsize()
			if _num_put > 0:
				with self.cache_lck:
					_num_cache = len(self.cache)
				if _num_cache > 0:
					_num_put = min(_num_put, _num_cache)
					if _num_put > 0:
						with self.cache_lck:
							_ = self.cache[:_num_put]
							self.cache = self.cache[_num_put:]
						for _nbatch in _:
							self.out.put(_nbatch)
							_ = None

	def loader(self):

		self.cache = []
		self.cache_lck = Lock()
		self.t_builder = start_thread(target=thread_keeper, args=((self.is_running,), all, self.sleep_secs,), kwargs={"target": self.builder})
		self.t_sender = start_thread(target=thread_keeper, args=((self.is_running,), all, self.sleep_secs,), kwargs={"target": self.sender})

	def is_running(self):

		return self.running.value

	def __call__(self, *args, **kwargs):

		while self.running.value:
			for _ in range(self.out.qsize()):
				if self.out.empty():
					break
				else:
					_ = self.out.get()
					yield from _

	def status(self, mode=True):

		self.running.value = 1 if mode else 0

	def close(self):

		self.running.value = 0
		while not self.out.empty():
			self.out.get()
