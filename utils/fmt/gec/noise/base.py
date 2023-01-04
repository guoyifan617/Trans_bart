#encoding: utf-8

from random import choices, randint, sample, shuffle

from utils.math import cumsum, pos_norm
from utils.vocab.char import ldvocab_list

from cnfg.vocab.plm.custbert import init_token_id, vocab_size

def load_replace_data(fname):

	rsd = {}
	with open(fname, "rb") as f:
		for _ in f:
			tmp = _.strip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if len(tmp) > 1:
					for _c in tmp:
						rsd[_c] = tmp

	return rsd

def filter_bi_same(samples, uni):

	_a, _b = samples

	return _b if _a == uni else _a

class NoiserBase:

	def edit(self, x, **kwargs):

		return x

	def __call__(self, x, sind, k, **kwargs):

		_eind = sind + k

		return "%s%s%s" % (x[:sind], self.edit(x[sind:_eind], **kwargs), x[_eind:])

class CharReplacer(NoiserBase):

	def __init__(self, df, sample_func=sample):

		self.rpd = load_replace_data(df)
		self.sample_func = sample_func

	def edit(self, x, sample_func=None, data=None):

		_sample_func, _rpd = self.sample_func if sample_func is None else sample_func, self.rpd if data is None else data
		rs = []
		for _ in x:
			rs.append(filter_bi_same(_sample_func(_rpd[_], 2), _) if _ in _rpd else _)

		return "".join(rs)

class VocabReplacer(NoiserBase):

	def __init__(self, df, vsize=vocab_size-init_token_id, sample_func=sample):

		self.rpd = ldvocab_list(df)
		self.sample_func = sample_func

	def edit(self, x, sample_func=None, data=None):

		_sample_func = self.sample_func if sample_func is None else sample_func
		_src_s = set(x)
		_src_len = len(x)
		rs = [_ for _ in _sample_func(self.rpd if data is None else data, _src_len + len(_src_s)) if _ not in _src_s]

		return "".join(rs[:_src_len])

class Shuffler(NoiserBase):

	def edit(self, x, **kwargs):

		_ = list(x)
		shuffle(_)

	return "".join(_)

class Repeater(NoiserBase):

	def edit(self, x, **kwargs):

		return "%s%s" % (x, x,)

def drop(self, x, sind, k, **kwargs):

	return "%s%s" % (x[:sind], x[sind + k:])

class Noiser:

	def __init__(self, char=None, vcb=None, min_span_len=1, max_span_len=5, p=0.15, w_char=0.3, w_vcb=0.1, w_shuf=0.2, w_repeat=0.1, w_drop=0.1):

		self.edits = []
		w = []
		self.inc_ind = self.dec_ind = self.shuf_ind = None
		if char is not None:
			if isinstance(char, str):
				self.edits.append(CharReplacer(char))
				w.append(w_char)
			else:
				self.edits.extend([CharReplacer(_) for _ in char])
				w.extend(w_char if isinstance(w_char, list) else [w_char for _ in range(len(char))])
				self.ind_shift.extend([0] for _ in range(len(char)))
		if vcb is not None:
			self.edits.append(VocabReplacer(vcb))
			w.append(w_vcb)
		if w_shuf > 0.0:
			self.shuf_ind = len(self.edits)
			self.edits.append(Shuffler())
			w.append(w_shuf)
		if w_repeat > 0.0:
			self.inc_ind = len(self.edits)
			self.edits.append(Repeater())
			w.append(w_repeat)
		if w_drop > 0.0:
			self.dec_ind = len(self.edits)
			self.edits.append(drop)
			w.append(w_drop)
		self.sample_cw = cumsum(pos_norm(w))
		self.sample_ind = list(range(len(self.edits)))
		self.min_span_len, self.max_span_len, self.p = min_span_len, max_span_len, p

	def __call__(self, x, **kwargs):

		_min_span_len, _max_span_len, _sample_ind, _sample_cw, _inc_ind, _dec_ind, _shuf_ind = self.min_span_len, self.max_span_len, self.sample_ind, self.sample_cw, self.inc_ind, self.dec_ind, self.shuf_ind
		_r_len = len(x)
		_last_ind = _r_len - 1
		_ = int(_r_len * self.p)
		_min_span_len, _max_span_len = max(min(_min_span_len, _), 1), max(min(_max_span_len, _), 1)
		_sind = 0
		_spans = []
		while _r_len > 0:
			_span_len = 1 if _max_span_len == 1 else min(randint(_min_span_len, _max_span_len), _r_len)
			_spans.append((_sind, _span_len,))
			_sind += _span_len
			_r_len -= _span_len
		_shift = 0
		rs = x
		for _sind, _span_len in choices(_spans, k=max(1, int(len(_spans) * self.p))):
			_ind = choices(_sample_ind, cum_weights=_sample_cw, k=1)[0]
			_r_sind = _sind + _shift
			if (_ind == _shuf_ind) and (_span_len == 1):
				_span_len = 2
				if _sind == _last_ind:
					_r_sind -= 1
			rs = self.edits[_ind](rs, _r_sind, _span_len, **kwargs)
			if _ind == _inc_ind:
				_shift += _span_len
			elif _ind == _dec_ind:
				_shift -= _span_len

		return rs