#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") and (fname.find("insert_arg") < 0)

def apply_patch(line, rules):

	patched = 0
	rs = line
	for fstr, src, tgt in rules:
		_ = rs.find(fstr)
		_lt = len(tgt)
		_ls = len(src)
		while _ >= 0:
			_rind = rs[_:].find(src)
			if _rind >= 0:
				_rind += _
				rs = rs[:_rind] + tgt + rs[_rind:]
				_sind = _rind + _lt + _ls
				_ = rs[_sind:].find(fstr)
				if _ >= 0:
					_ += _sind
				patched += 1

	return patched, rs

def patch_decoder(fname):

	patched = 0

	prules = (("__insert_start", "__insert_end", "__insert__tgt",),)

	cache = []
	prev_emp = False
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.rstrip()
			if tmp:
				_patched, tmp = apply_patch(tmp.decode("utf-8"), prules)
				patched += _patched
				tmp = tmp.rstrip()
				if tmp:
					cache.append(tmp)
					prev_emp = False
			else:
				if not prev_emp:
					cache.append(tmp.decode("utf-8"))
				prev_emp = True
	while not cache[-1]:
		del cache[-1]
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write("\n".encode("utf-8"))

	return patched

def walk_path(ptws):

	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				if patch_decoder(ptw) > 0:
					print("Patched %s" % (ptw))
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						#print("Processing %s" % (_execf))
						if patch_decoder(_execf) > 0:
							print("Patched %s" % (_execf))

if __name__ == "__main__":
	walk_path(sys.argv[1:])
