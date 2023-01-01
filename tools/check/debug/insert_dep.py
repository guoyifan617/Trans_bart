#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

depl = ["eos_id", "pad_id"]
indep = "from cnfg.vocab.base import "

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("insert_dep") < 0)

def insert_dep(lin, depl, depc):

	do_ins = False
	do_ins_d = [False for dep in depl]
	has_package = False
	deps = set()
	for lu in lin:
		if lu.find(depc) >= 0:
			has_package = True
			if lu.find("*") > 0:
				do_ins = False
				break
			else:
				_ = lu.replace(depc, "")
				_tmp = True
				for _d in depl:
					_c = _.find(_d) >= 0
					if _c and (_d not in deps):
						deps.add(_d)
					if not _c:
						_tmp = False
				if _tmp:
					do_ins = False
					break
		else:
			_ind = lu.find("#")
			_lu = lu[:_ind] if _ind >= 0 else lu
			for ind, dep in enumerate(depl):
				if (dep not in deps) and (_lu.find(dep) >= 0):
					do_ins = True
					do_ins_d[ind] = True
	if do_ins:
		rs = []
		_not_imported = True
		tag = [dep for dep, b in zip(depl, do_ins_d) if b]
		aip = (", " + ", ".join(tag)) if has_package else ("\n" + depc + ", ".join(tag) + "\n")
		for lu in lin:
			tmp = lu
			_tmp = tmp.strip()
			if _tmp:
				if _not_imported:
					if has_package:
						if tmp.find(depc) >= 0:
							tmp += aip
							_not_imported = False
					elif (not (_tmp.startswith("#") or _tmp.find("import") >= 0)):
						rs.append(aip)
						_not_imported = False
			rs.append(tmp)
		return rs
	else:
		return lin

def process_file(fname, depl, depc):

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
	cache = insert_dep(cache, depl, depc)
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

def walk_path(ptws):

	global depl, indep
	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				process_file(ptw, depl, indep)
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						process_file(_execf, depl, indep)

if __name__ == "__main__":
	walk_path(sys.argv[1:])
