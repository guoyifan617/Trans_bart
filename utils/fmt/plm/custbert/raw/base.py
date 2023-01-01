#encoding: utf-8

from random import shuffle

from utils.fmt.base import FileList, read_lines
from utils.fmt.plm.custbert.raw.single.char import doc_file_reader, sent_file_reader

def inf_file_loader(sfiles, dfiles, max_len=510, sent_file_reader=sent_file_reader, doc_file_reader=doc_file_reader, print_func=print):

	with FileList(sfiles, "rb") as _s_files, FileList(dfiles, "rb") as _d_files:
		while True:
			_fnames = sfiles + dfiles
			for _ in _s_files:
				_.seek(0)
			for _ in _d_files:
				_.seek(0)
			_files = [sent_file_reader(_, max_len=max_len) for _ in _s_files]
			_files.extend([doc_file_reader(_, max_len=max_len) for _ in _d_files])
			if print_func is not None:
				for _ in _fnames:
					print_func("open %s" % _)
			while _files:
				_cl = []
				for i, _f in enumerate(_files):
					_data = next(_f, None)
					if _data is None:
						_cl.append(i)
					else:
						yield _data
				if _cl:
					for _ in reversed(_cl):
						del _files[_]
						if print_func is not None:
							print_func("close %s" % _fnames.pop(_))

def sort_list_file_reader(x, *args, clear_input=True, **kwargs):

	_d = {}
	for _ in x:
		_k = len(_)
		if _k in _d:
			if _ not in _d[_k]:
				_d[_k].add(_)
		else:
			_d[_k] = set([_])
	if clear_input:
		x.clear()
	for _k in sorted(_d.keys()):
		_v = list(_d.pop(_k))
		shuffle(_v)
		yield from _v

class sort_lines_reader:

	def __init__(self, line_read=None):

		self.line_read = line_read

	def __call__(self, x, *args, line_read=None, **kwargs):

		_line_read = self.line_read if line_read is None else line_read
		_data_iter = x if _line_read is None else read_lines(x, _line_read)
		_d = {}
		for _ in _data_iter:
			_k = len(_)
			if _k in _d:
				if _ not in _d[_k]:
					_d[_k].add(_)
			else:
				_d[_k] = set([_])
		for _k in sorted(_d.keys()):
			_v = list(_d.pop(_k))
			shuffle(_v)
			yield from _v
