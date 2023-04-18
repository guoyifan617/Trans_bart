#encoding: utf-8

from utils.fmt.base import FileList
from utils.fmt.parser import parse_none

from cnfg.hyp import cache_len_default
from cnfg.vocab.gector.det import correct_id, incorrect_id

def gec_noise_reader_core(files=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, **kwargs):

	_s_max_len = max_len - 2
	for _f in files:
		for line in _f:
			tgt = line.strip()
			if tgt:
				tgt = tgt.decode("utf-8")
				_l = len(tgt)
				if (_l > min_len) and (_l < _s_max_len):
					src = noiser(tgt)
					_l = len(src)
					if (src != tgt) and (_l > min_len) and (_l < _s_max_len):
						yield tuple(tokenizer(tgt)), tuple([correct_id])
						yield tuple(tokenizer(src)), tuple([incorrect_id])

def gec_noise_reader(fname=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, inf_loop=False, **kwargs):

	with FileList([fname] if isinstance(fname, str) else fname, "rb") as files:
		if inf_loop:
			while True:
				for _ in files:
					_.seek(0)
				for _ in gec_noise_reader_core(files=files, noiser=noiser, tokenizer=tokenizer, min_len=min_len, max_len=max_len):
					yield _
		else:
			for _ in gec_noise_reader_core(files=files, noiser=noiser, tokenizer=tokenizer, min_len=min_len, max_len=max_len):
				yield _

class GECNoiseReader:

	def __init__(self, fname, noiser, tokenizer, min_len=2, max_len=cache_len_default, inf_loop=False, **kwargs):

		self.fname, self.noiser, self.tokenizer, self.min_len, self.max_len, self.inf_loop = fname, noiser, tokenizer, min_len, max_len, inf_loop

	def __call__(self, fname=None, noiser=None, tokenizer=None, min_len=None, max_len=None, inf_loop=None, **kwargs):

		return gec_noise_reader(fname=parse_none(fname, self.fname), noiser=parse_none(noiser, self.noiser), tokenizer=parse_none(tokenizer, self.tokenizer), min_len=parse_none(min_len, self.min_len), max_len=parse_none(max_len, self.max_len), inf_loop=parse_none(inf_loop, self.inf_loop), **kwargs)
