#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("fix_import") < 0)

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

def load_defs(lin):

	rs = set()
	_try_except_mode = False
	for tmp in skip_comment(lin):
		if tmp:
			_tmp_s = tmp.strip()
			_is_no_ident_line = not (tmp.startswith("\t") or tmp.startswith(" "))
			if _is_no_ident_line or _try_except_mode:
				_ = tmp.find("#")
				if _ >= 0:
					tmp = tmp[:_]
				if tmp:
					if _is_no_ident_line:
						_try_except_mode = False
					if tmp.startswith("def "):
						_ = tmp.find("(")
						if _ > 4:
							_m = tmp[4:_].strip()
							if legal_name(_m) and (_m not in rs):
								rs.add(_m)
					elif tmp.startswith("class "):
						_ = tmp.find("(")
						if _ < 0:
							_ = tmp.find(":")
						if _ > 6:
							_m = tmp[6:_].strip()
							if legal_name(_m) and (_m not in rs):
								rs.add(_m)
					elif _tmp_s.startswith("from "):
						_ = tmp.find(" import ")
						if _ > 5:
							for tmp in tmp[_ + 8:].split(","):
								_ = tmp.find(" as ")
								_m = tmp[_ + 4:] if _ > 0 else tmp
								_m = _m.strip()
								if legal_name(_m) and (_m not in rs):
									rs.add(_m)
					elif tmp.startswith("try:") or tmp.startswith("except "):
						_try_except_mode = True
					else:
						_ = tmp.find("=")
						if (_ > 0) and (tmp[_ + 1] != "="):
							tmp = [_t.strip() for _t in tmp[:_].strip().split(",")]
							tmp = [_t for _t in tmp if _t]
							if all(legal_name(_) for _ in tmp):
								for _m in tmp:
									if _m not in rs:
										rs.add(_m)

	return rs

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

def fix_import_core(lin, impl, cts):

	_fix = False
	_c_defs = cts - load_defs(lin)
	for lu in lin:
		if lu.strip() == impl:
			_fix = True
			break
	if _fix:
		_deps = set()
		for lu in skip_comment(lin):
			for _ in (set([_t for _t in lu.strip().replace("~", " ").replace("+", " ").replace("-", " ").replace("*", " ").replace("/", " ").replace("%", " ").replace("(", " ").replace(")", " ").replace("[", " ").replace("]", " ").replace("{", " ").replace("}", " ").replace("=", " ").replace(",", " ").replace(".", " ").replace(":", " ").replace("@", " ").split() if _t]) & _c_defs):
				if _ not in _deps:
					_deps.add(_)
		_deps = list(_deps)
		if _deps:
			_deps.sort()
			_deps = impl[:-1] + ", ".join(_deps)
			return [lu.replace(impl, _deps) if lu.strip() == impl else lu for lu in lin], _fix
		else:
			return [lu for lu in lin if lu.strip() != impl], _fix
	else:
		return lin, _fix

def process_file(fname, impl, cts):

	cache = load_file(fname)
	cache, _fix = fix_import_core(cache, impl, cts)
	if _fix:
		print(fname)
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

def handle(srcf, ptws):

	_defs = load_defs(load_file(srcf))
	_impl = "from %s import *" % srcf.replace("/", ".")[:-3]
	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				process_file(ptw, _impl, _defs)
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						process_file(_execf, _impl, _defs)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2:])
