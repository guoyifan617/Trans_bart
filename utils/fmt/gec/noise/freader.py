#encoding: utf-8

from utils.fmt.base import line_reader
from utils.fmt.gec.gector.base import generate_iter_data
from utils.fmt.parser import parse_none

from cnfg.hyp import cache_len_default

def gec_noise_reader(fname=None, noiser=None, tokenizer=None, min_len=2, max_len=cache_len_default, print_func=print):

	_s_max_len = max_len - 2
	_fname = [fname] if isinstance(fname, str) else fname
	for _f in _fname:
		for tgt in line_reader(_f, keep_empty_line=False, print_func=print_func):
			_l = len(tgt)
			if (_l > min_len) and (_l < _s_max_len):
				src = noiser(tgt)
				for _s, _e, _t in generate_iter_data(tokenizer(src), tokenizer(tgt)):
					_l = len(_s)
					if _l < max_len:
						yield tuple(_s), tuple(_e), tuple(_t)

class GECNoiseReader:

	def __init__(self, fname, noiser, tokenizer, min_len=2, max_len=cache_len_default, print_func=print):

		self.fname, self.noiser, self.tokenizer, self.min_len, self.max_len, self.print_func = fname, noiser, tokenizer, min_len, max_len, print_func

	def __call__(self, fname=None, noiser=None, tokenizer=None, min_len=None, max_len=None, print_func=print):

		return gec_noise_reader(fname=parse_none(fname, self.fname), noiser=parse_none(noiser, self.noiser), tokenizer=parse_none(tokenizer, self.tokenizer), min_len=parse_none(min_len, self.min_len), max_len=parse_none(max_len, self.max_len), print_func=parse_none(print_func, self.print_func))
