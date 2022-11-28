#encoding: utf-8

import sys
from os import walk
from os.path import join as pjoin

indep = "from utils.h5serial import "
depafter = "from utils."

def insert_dep(lin):

	global indep, depafter
	exist_indep = exist_depafter = False
	_c_depc = indep.strip()
	for lu in lin:
		if lu.find(indep) >= 0:
			exist_indep = True
			depinsert = lu
		elif lu.find(depafter) >= 0:
			exist_depafter = True
	not_inserted = True
	if exist_depafter and exist_indep:
		rs = []
		for lu in lin:
			_tmp = lu.strip()
			insert_lu = True
			if _tmp:
				if not_inserted and lu.find(depafter) >= 0 and lu.find(indep) < 0:
					rs.append(lu)
					rs.append(depinsert)
					insert_lu = False
					not_inserted = False
				elif lu.find(indep) >= 0:
					insert_lu = False
			if insert_lu:
				rs.append(lu)
		return rs
	else:
		return lin

def process_file(fname):

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
	cache = insert_dep(cache)
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

def walk_path(ptw):

	for root, dirs, files in walk(ptw):
		for execf in files:
			if execf.endswith(".py") and execf.find("update_dep") < 0 and execf.find("debug") < 0:
				_execf = pjoin(root, execf)
				process_file(_execf)

if __name__ == "__main__":
	walk_path(sys.argv[1])
