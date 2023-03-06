#encoding: utf-8

import torch
from multiprocessing import Manager, Value
from numpy import array as np_array, int32 as np_int32
from os.path import exists as fs_check
from random import seed as rpyseed, shuffle
from shutil import rmtree
from time import sleep
from uuid import uuid4 as uuid_func

from utils.fmt.plm.custbert.raw.base import inf_file_loader
from utils.fmt.raw.cachepath import get_cache_fname, get_cache_path
from utils.fmt.raw.reader.sort.single import sort_lines_reader
from utils.fmt.single import batch_padder
from utils.fmt.vocab.char import ldvocab
from utils.fmt.vocab.plm.custbert import map_batch
from utils.h5serial import h5File
from utils.process import start_process

from cnfg.base import seed as rand_seed
from cnfg.ihyp import h5_libver, h5datawargs, max_pad_tokens_sentence, max_sentences_gpu, max_tokens_gpu, normal_tokens_vs_pad_tokens
from cnfg.vocab.plm.custbert import init_normal_token_id, init_vocab, pad_id, vocab_size

class Loader:

	def __init__(self, sfiles, dfiles, vcbf, max_len=510, num_cache=4, raw_cache_size=4194304, skip_lines=0, nbatch=256, minfreq=False, vsize=vocab_size, ngpu=1, bsize=max_sentences_gpu, maxpad=max_pad_tokens_sentence, maxpart=normal_tokens_vs_pad_tokens, maxtoken=max_tokens_gpu, sleep_secs=1.0, file_loader=inf_file_loader, ldvocab=ldvocab, print_func=print):

		self.sent_files, self.doc_files, self.max_len, self.num_cache, self.raw_cache_size, self.skip_lines, self.nbatch, self.minbsize, self.maxpad, self.maxpart, self.sleep_secs, self.file_loader, self.print_func = sfiles, dfiles, max_len, num_cache, raw_cache_size, skip_lines, nbatch, ngpu, maxpad, maxpart, sleep_secs, file_loader, print_func
		self.bsize, self.maxtoken = (bsize, maxtoken,) if self.minbsize == 1 else (bsize * self.minbsize, maxtoken * self.minbsize,)
		self.cache_path = get_cache_path(*self.sent_files, *self.doc_files)
		self.vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
		self.manager = Manager()
		self.out = self.manager.list()
		self.todo = self.manager.list([get_cache_fname(self.cache_path, i=_) for _ in range(self.num_cache)])
		self.running = Value("B", 1, lock=True)
		self.p_loader = start_process(target=self.loader)
		self.iter = None

	def loader(self):

		rpyseed(rand_seed)
		dloader = self.file_loader(self.sent_files, self.doc_files, max_len=self.max_len, print_func=None)
		file_reader = sort_lines_reader(line_read=self.raw_cache_size)
		if self.skip_lines > 0:
			_line_read = self.skip_lines - 1
			for _ind, _ in enumerate(dloader, 1):
				if _ind > _line_read:
					break
		while self.running.value:
			if self.todo:
				_cache_file = self.todo.pop(0)
				with h5File(_cache_file, "w", libver=h5_libver) as rsf:
					src_grp = rsf.create_group("src")
					curd = 0
					for i_d in batch_padder(dloader, self.vcb, self.bsize, self.maxpad, self.maxpart, self.maxtoken, self.minbsize, file_reader=file_reader, map_batch=map_batch, pad_id=pad_id):
						src_grp.create_dataset(str(curd), data=np_array(i_d, dtype=np_int32), **h5datawargs)
						curd += 1
					rsf["ndata"] = np_array([curd], dtype=np_int32)
				self.out.append(_cache_file)
			else:
				sleep(self.sleep_secs)

	def iter_func(self, *args, **kwargs):

		while self.running.value:
			if self.out:
				_cache_file = self.out.pop(0)
				if fs_check(_cache_file):
					try:
						td = h5File(_cache_file, "r")
					except Exception as e:
						td = None
						if self.print_func is not None:
							self.print_func(e)
					if td is not None:
						if self.print_func is not None:
							self.print_func("load %s" % _cache_file)
						tl = [str(i) for i in range(td["ndata"][()].item())]
						shuffle(tl)
						src_grp = td["src"]
						for i_d in tl:
							yield torch.from_numpy(src_grp[i_d][()])
						td.close()
						if self.print_func is not None:
							self.print_func("close %s" % _cache_file)
				self.todo.append(_cache_file)
			else:
				sleep(self.sleep_secs)

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
		sleep(self.sleep_secs)
		rmtree(self.cache_path, ignore_errors=True)
