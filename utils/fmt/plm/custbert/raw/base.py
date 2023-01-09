#encoding: utf-8

from utils.fmt.base import FileList
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
