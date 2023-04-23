#encoding: utf-8

import sys
from os import walk
from os.path import exists as fs_check, isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("sort_deps") < 0)

def legal_name(strin):

	return (strin.find(" ") < 0) and (strin.find("\t") < 0) and (strin != "*")

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

def is_cust_py(impl):

	_ = impl.strip()
	_find = _.find("from ")
	_ind = _.find("import ")
	if _find >= 0:
		libname = _[_find + 5:_ind].strip()
	elif _ind >= 0:
		libname = _[_ind + 7:].strip()
	else:
		return True
	_ind = libname.find(".")
	libname = libname[:_ind] if _ind > 0 else ("%s.py" % libname)

	return fs_check(libname)

def sort_comp_core(lin):

	rs = [_t.strip() for _t in lin.split(",")]
	rs = [_t for _t in rs if _t]
	rs.sort()

	return ", ".join(rs)

def sort_comp(impl):

	_ind = impl.find(" import ") + 8
	rs = [sort_comp_core(_) for _ in impl[_ind:].split("#")]

	return "%s%s" % (impl[:_ind], "#, ".join(rs),)

def is_import_line(strin):

	return ((not strin.strip()) or strin.find("import ") >= 0) and (strin.find("\"") < 0) and (strin.find("'") < 0) and (not strin.strip().startswith("# "))

def sort_imp(lin):

	_ = {}
	for lu in lin:
		_k = lu.strip()
		_ind = _k.find("#")
		if _ind >= 0:
			_k = _k[:_ind]
		_k = _k[7:].strip()
		if _k in _:
			print("unable to handle repeated import %s" % _k)
			exit()
		_[_k] = lu

	return [_[_k] for _k in sorted(_.keys())]

def merge_frm_imp(la, lb):

	_ind = la.find(" import ") + 8
	_a = la[_ind:].strip().split("#")
	_b = lb[lb.find(" import ") + 8:].strip().split("#")
	_ = "%s%s, %s" % (la[:_ind], _a[0], _b[0])
	_c = _a[1:] + _b[1:]

	return sort_comp(("%s$, %s" % (_, "#, ".join(_c))) if _c else _)

def sort_frm(lin):

	_ = {}
	for lu in lin:
		_t = sort_comp(lu)
		_k = _t[_t.find("from ") + 5:_t.find(" import ")].strip()
		if _k in _:
			_t = merge_frm_imp(_[_k], _t)
			print("merge (from) import \"%s\" and \"%s\" into \"%s\"" % (_[_k], lu, _t,))
			_[_k] = _t
			exit()
		else:
			_[_k] = _t

	return [_[_k] for _k in sorted(_.keys())]

def sort_dep_lines(lin):

	sys_py_imp = []
	sys_py_frm = []
	cust_py_imp = []
	cust_py_frm = []
	cnfg_py_imp = []
	cnfg_py_frm = []
	for lu in lin:
		if lu.strip():
			_is_from_imp = lu.find("from ") >= 0
			if (lu.find(" cnfg.") >= 0) or (lu.find(" cnfg ") >= 0) or ((lu.find(" cnfg") >= 0) and (not _is_from_imp)):
				if _is_from_imp:
					cnfg_py_frm.append(lu)
				else:
					cnfg_py_imp.append(lu)
			elif is_cust_py(lu):
				if _is_from_imp:
					cust_py_frm.append(lu)
				else:
					cust_py_imp.append(lu)
			else:
				if _is_from_imp:
					sys_py_frm.append(lu)
				else:
					sys_py_imp.append(lu)
	rs = [""]
	if sys_py_imp:
		rs.extend(sort_imp(sys_py_imp))
	if sys_py_frm:
		rs.extend(sort_frm(sys_py_frm))
	if rs[-1] != "":
		rs.append("")
	if cust_py_imp:
		rs.extend(sort_imp(cust_py_imp))
	if cust_py_frm:
		rs.extend(sort_frm(cust_py_frm))
	if rs[-1] != "":
		rs.append("")
	if cnfg_py_imp:
		rs.extend(sort_imp(cnfg_py_imp))
	if cnfg_py_frm:
		rs.extend(sort_frm(cnfg_py_frm))
	if rs[-1] != "":
		rs.append("")

	return [rs[1]] if (len(rs) == 3) and (len(lin) == 1) else rs

def sort_deps(lin):

	rs = []
	deps_cache = []
	for lu in lin:
		if is_import_line(lu):
			deps_cache.append(lu)
		else:
			if deps_cache:
				rs.extend(sort_dep_lines(deps_cache))
				deps_cache = []
			rs.append(lu)
	if deps_cache:
		rs.extend(sort_dep_lines(deps_cache))

	return rs

def process_file(fname):

	print(fname)
	cache = load_file(fname)
	cache = sort_deps(cache)
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

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
