#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("redeps") < 0)

def legal_name(strin):

	return (strin.find(" ") < 0) and (strin.find("\t") < 0) and (strin != "*")

def skip_comment(lin):

	_comment = False
	rs = set()
	for tmp in lin:
		if tmp:
			_tmp_s = tmp.strip()
			if _tmp_s.find("\"\"\"") >= 0:
				_comment = not _comment
				if not _comment:
					continue
			if not _comment:
				_ = tmp.find("#")
				if _ >= 0:
					tmp = tmp[:_]
				if tmp:
					yield tmp

def load_file(fname):

	cache = []
	prev_emp = False
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.rstrip()
			if tmp:
				cache.append(tmp.decode("utf-8"))
				prev_emp = False
			else:
				if not prev_emp:
					cache.append(tmp.decode("utf-8"))
				prev_emp = True
	while not cache[-1]:
		del cache[-1]

	return cache

def get_all_names(lin):

	_names = set()
	for lu in skip_comment(lin):
		if lu.find("import ") < 0:
			_names |= set([_t for _t in lu.strip().replace("~", " ").replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("%", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace("=", " ").replace(",", " ").replace(".", " ").replace(":", " ").replace("@", " ").split() if _t])

	return _names

def get_imports(lin):

	_imps = set()
	for lu in skip_comment(lin):
		_ind = lu.find("import ")
		if _ind >= 0:
			for tmp in lu[_ind + 7:].strip().split(","):
				tmp = tmp.strip()
				if tmp:
					_ = tmp.find(" as ")
					_m = tmp[_ + 4:] if _ > 0 else tmp
					_m = _m.strip()
					if legal_name(_m) and (_m not in _imps):
						_imps.add(_m)

	return _imps

def process_file(fname):

	cache = load_file(fname)
	_all_names = get_all_names(cache)
	_imps = get_imports(cache)
	_ = _imps - _all_names
	if _:
		print(fname)
		print(_)

def handle(ptws):

	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				process_file(ptw)
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						process_file(_execf)

if __name__ == "__main__":
	handle(sys.argv[1:])
