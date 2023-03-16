#encoding: utf-8

from difflib import Differ, SequenceMatcher

default_differ = False

op_mapper = {" ": "k", "+": "i", "-": "d"}

differ = Differ()

diff_func_differ = differ.compare

def diff_func_matcher(x, ref):

	rs = []
	for tag, xsi, xei, rsi, rei in SequenceMatcher(None, x, ref, autojunk=False).get_opcodes():
		_tc = tag[0]
		if _tc == "d":
			rs.extend("- %s" % _ for _ in x[xsi:xei])
		elif _tc == "e":
			rs.extend("  %s" % _c for _c in x[xsi:xei])
		elif _tc == "i":
			rs.extend("+ %s" % _c for _c in ref[rsi:rei])
		else:
			rs.extend("- %s" % _c for _c in x[xsi:xei])
			rs.extend("+ %s" % _c for _c in ref[rsi:rei])

	return rs

def diff_func_matcher_reorder_insert(x, ref):

	rs = []
	for tag, xsi, xei, rsi, rei in SequenceMatcher(None, x, ref, autojunk=False).get_opcodes():
		_tc = tag[0]
		if _tc == "d":
			rs.extend("- %s" % _ for _ in x[xsi:xei])
		elif _tc == "e":
			rs.extend("  %s" % _c for _c in x[xsi:xei])
		elif _tc == "i":
			rs.extend("+ %s" % _c for _c in ref[rsi:rei])
		else:
			rs.extend("+ %s" % _c for _c in ref[rsi:rei])
			rs.extend("- %s" % _c for _c in x[xsi:xei])

	return rs

diff_func = diff_func_differ if default_differ else diff_func_matcher

class TokenMapper:

	def __init__(self, map_t=None, map_c=None, tid=1):

		self.map_t = {} if map_t is None else map_t
		self.map_c = {} if map_c is None else map_c
		self.tid = tid

	def map(self, *args):

		rs = []
		for _seq in args:
			_seq_rs = []
			for _tok in _seq:
				if _tok in self.map_t:
					_seq_rs.append(self.map_t[_tok])
				else:
					_c = chr(self.tid)
					_seq_rs.append(_c)
					self.map_t[_tok] = _c
					self.map_c[_c] = _tok
					self.tid += 1
			rs.append(_seq_rs)

		return rs

	def map_back(self, *args):

		return [[self.map_c[_] for _ in _seq] for _seq in args]

def seq_diff(a, b, op_mapper=op_mapper, diff_func=diff_func):

	_mapper = TokenMapper()
	_ma, _mb = _mapper.map(a, b)
	_map_c = _mapper.map_c
	for _ in diff_func(_ma, _mb):
		yield op_mapper[_[0]], _map_c[_[-1]]

def reorder_insert(seqin):

	_d_cache = []
	for _du in seqin:
		_op = _du[0]
		if _op == "d":
			_d_cache.append(_du)
		else:
			if (_op == "k") and _d_cache:
				yield from _d_cache
				_d_cache = []
			yield _du
	if _d_cache:
		yield from _d_cache

def seq_diff_reorder_insert_differ(a, b, op_mapper=op_mapper):

	return reorder_insert(seq_diff(a, b, op_mapper=op_mapper, diff_func=diff_func_differ))

def seq_diff_reorder_insert_matcher(a, b, op_mapper=op_mapper):

	return seq_diff(a, b, op_mapper=op_mapper, diff_func=diff_func_matcher_reorder_insert)

seq_diff_reorder_insert = seq_diff_reorder_insert_differ if default_differ else seq_diff_reorder_insert_matcher
