#encoding: utf-8

from utils.fmt.base import line_reader
from utils.fmt.gec.gector.base import generate_iter_data
from utils.fmt.parser import parse_none

from cnfg.hyp import cache_len_default

class gec_noise_reader:

	def __init__(self, fname, noiser, tokenizer, min_len=2, max_len=cache_len_default, print_func=print):

		self.fname, self.noiser, self.tokenizer, self.min_len, self.max_len, self.print_func = fname, noiser, tokenizer, min_len, max_len, print_func

	def __call__(self, fname=None, noiser=None, tokenizer=None, min_len=None, max_len=None, print_func=print):

		_fname, _noiser, _token, _min_len, _max_len = parse_none(fname, self.fname), parse_none(noiser, self.noiser), parse_none(tokenizer, self.tokenizer), parse_none(min_len, self.min_len), parse_none(max_len, self.max_len)
		_s_max_len = _max_len - 2
		_fname = [_fname] if isinstance(_fname, str) else _fname
		for _f in _fname:
			for tgt in line_reader(_f, keep_empty_line=False, print_func=parse_none(print_func, self.print_func)):
				_l = len(tgt)
				if (_l > _min_len) and (_l < _s_max_len):
					src = _noiser(tgt)
					for _s, _e, _t in generate_iter_data(_token(src), _token(tgt)):
						_l = len(_s)
						if _l < _max_len:
							yield _s, _e, _t
