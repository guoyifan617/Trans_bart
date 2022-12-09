#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("sep_deps") < 0)

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

def load_defs(lin, skip_package=None):

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
						if (skip_package is None) or _tmp_s.find(" %s " % skip_package) < 0:
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

def sep_import_core(lin, oimpl, nimpl, cts):

	_fix = False
	for lu in lin:
		_ = lu.strip()
		if _.startswith(oimpl):
			_oc = _[len(oimpl):]
			if _oc:
				tmp = _oc.split(",")
				tmp = set([_.strip() for _ in tmp])
				_nc = tmp & cts
				if _nc:
					_fix = True
					_oc = tmp - cts
					_head = lu[:lu.find(" import ") + 8]
					if _oc:
						_imp = "%s%s\n%s%s" % (_head, ", ".join(sorted(list(_oc))), _head.replace(oimpl, nimpl), ", ".join(sorted(list(_nc))),)
					else:
						_imp = _head.replace(oimpl, nimpl) + ", ".join(sorted(list(_nc)))
	if _fix:
		return [_imp if lu.strip().startswith(oimpl) else lu for lu in lin], _fix
	else:
		return lin, _fix

def process_file(fname, oimpl, nimpl, cts):

	cache = load_file(fname)
	cache, _fix = sep_import_core(cache, oimpl, nimpl, cts)
	if _fix:
		print(fname)
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

def handle(srcf, opkg, ptws):

	_defs = load_defs(load_file(srcf), skip_package=opkg)
	_oimpl = "from %s import " % opkg
	_nimpl = "from %s import " % srcf.replace("/", ".")[:-3]
	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				process_file(ptw, _oimpl, _nimpl, _defs)
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						process_file(_execf, _oimpl, _nimpl, _defs)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3:])
