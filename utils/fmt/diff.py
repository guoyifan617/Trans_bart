#encoding: utf-8

from difflib import Differ

differ = Differ()

diff_func = differ.compare

op_mapper = {" ": "k", "+": "i", "-": "d"}

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

def seq_diff(a, b, op_mapper=op_mapper):

	_mapper = TokenMapper()
	_ma, _mb = _mapper.map(a, b)
	_map_c = _mapper.map_c
	rs = []
	for _ in diff_func(_ma, _mb):
		rs.append((op_mapper[_[0]], _map_c[_[-1]],))

	return rs

def reorder_insert(seqin):

	_d_cache = []
	rs = []
	for _du in seqin:
		_op = _du[0]
		if _op == "d":
			_d_cache.append(_du)
		else:
			if (_op == "k") and _d_cache:
				rs.extend(_d_cache)
				_d_cache = []
			rs.append(_du)
	if _d_cache:
		rs.extend(_d_cache)
		_d_cache = []

	return rs

def seq_diff_reorder_insert(a, b, op_mapper=op_mapper):

	return reorder_insert(seq_diff(a, b, op_mapper=op_mapper))
