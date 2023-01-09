#encoding: utf-8

import torch
from multiprocessing import Queue, Value
from random import seed as rpyseed, shuffle
from threading import Lock
from time import sleep

from utils.fmt.base import seperate_list
from utils.fmt.plm.custbert.raw.base import inf_file_loader
from utils.fmt.raw.reader.sort.single import sort_lines_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.process import start_process
from utils.thread import start_thread

from cnfg.base import seed as rand_seed
from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=128, raw_cache_size=1048576, skip_lines=0, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.skip_lines, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, skip_lines, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.out = Queue()
		self.running = Value("B", 1, lock=True)
		self.p_loader = start_process(target=self.loader)
		self.t_builder = self.t_sender = None
		self.iter = None

	def builder(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=None)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		if self.skip_lines > 0:
			_line_read = self.skip_lines - 1
			for _ind, _ in enumerate(dloader, 1):
				if _ind > _line_read:
					break
		_cpu = torch.device("cpu")
		while self.running.value:
			with self.cache_lck:
				_num_cache = len(self.cache)
			if _num_cache < self.num_cache:
				# as the reference to the tensor will be released after put into the queue, we cannot move it to the shared memory with .share_memory_()
				_raw = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(dloader, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader, map_batch=map_batch, pad_id=pad_id)]
				shuffle(_raw)
				_raw = seperate_list(_raw, self.nbatch)
				with self.cache_lck:
					self.cache.extend(_raw)
				_raw = None
			else:
				sleep(self.sleep_secs)

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
			else:
				sleep(self.sleep_secs)

	def loader(self):

		rpyseed(rand_seed)
		self.cache = []
		self.cache_lck = Lock()
		self.t_builder = start_thread(target=self.builder)
		self.t_sender = start_thread(target=self.sender)

	def is_running(self):

		return self.running.value

	def iter_func(self, *args, **kwargs):

		while self.running.value:
			for _ in range(self.out.qsize()):
				if self.out.empty():
					break
				else:
					_ = self.out.get()
					yield from _

	def __call__(self, *args, **kwargs):

		if self.iter is None:
			self.iter = self.iter_func(*args, **kwargs)
		for _ in self.iter:
			yield _
		self.iter = None

	def status(self, mode=True):

		with self.running.get_lock():
			self.running.value = 1 if mode else 0

	def close(self):

		self.running.value = 0
		while not self.out.empty():
			self.out.get()
