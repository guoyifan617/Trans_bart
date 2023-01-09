#encoding: utf-8

import torch
from random import shuffle
from threading import Lock
from time import sleep

from utils.fmt.base import seperate_list
from utils.fmt.plm.custbert.raw.base import inf_file_loader
from utils.fmt.raw.reader.sort.single import sort_lines_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.thread import LockHolder, start_thread

from cnfg.ihyp import max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=128, raw_cache_size=1048576, skip_lines=0, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.skip_lines, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, skip_lines, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.out = []
		self.out_lck = Lock()
		self.running = LockHolder(True)
		self.t = start_thread(target=self.loader)
		self.iter = None

	def loader(self):

		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=self.print_func)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		if self.skip_lines > 0:
			_line_read = self.skip_lines - 1
			for _ind, _ in enumerate(dloader, 1):
				if _ind > _line_read:
					break
		_cpu = torch.device("cpu")
		while self.running():
			with self.out_lck:
				_num_out = len(self.out)
			if self.num_cache > _num_out:
				_cache = [torch.as_tensor(_, dtype=torch.int32, device=_cpu) for _ in batch_padder(dloader, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader, map_batch=map_batch, pad_id=pad_id)]
				shuffle(_cache)
				_cache = seperate_list(_cache, self.nbatch)
				with self.out_lck:
					self.out.extend(_cache)
				_cache = None
			else:
				sleep(self.sleep_secs)

	def iter_func(self, *args, **kwargs):

		while self.running():
			with self.out_lck:
				if len(self.out) > 0:
					_ = self.out.pop(0)
				else:
					_ = None
			if _ is not None:
				yield from _

	def __call__(self, *args, **kwargs):

		if self.iter is None:
			self.iter = self.iter_func()
		for _ in self.iter:
			yield _
		self.iter = None

	def status(self, mode=True):

		self.running(mode)

	def close(self):

		self.running(False)
		with self.out_lck:
			self.out.clear()
