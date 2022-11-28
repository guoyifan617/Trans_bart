#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def check_file_type(fname):

	return fname.endswith(".py") or fname.endswith(".sh") or fname.endswith(".md") or fname.endswith(".cpp") or fname.endswith(".h")

def clean_lspace(lin):

	rs = []
	prev_tab = prev_space = False
	for i, c in enumerate(lin):
		if c == "\t":
			prev_tab = True
		elif c == " ":
			if prev_tab:
				rs.append(lin[:i])
			else:
				prev_space = True
			prev_tab = False
		else:
			if prev_tab or prev_space:
				rs.append(lin[:i])
			rs.append(" ".join([tmpu for tmpu in lin[i:].split(" ") if tmpu]))
			break

	return "".join(rs)

def clean_tab(fname):

	cache = []
	prev_emp = False
	is_cpp_file = fname.endswith(".cpp") or fname.endswith(".h")
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.rstrip()
			if tmp:
				tmp = clean_lspace(tmp.decode("utf-8"))
				if is_cpp_file:
					tmp = tmp.replace("){", ") {").replace("else{", "else {")
				#_tmp_strip = tmp.lstrip()
				#if (_tmp_strip.startswith("class ") or _tmp_strip.startswith("def ")) and _tmp_strip[:_tmp_strip.rfind("#")].endswith(":"):
				#	if not prev_emp:
				#		cache.append("")
				#	cache.append(tmp)
				#	cache.append("")
				#	prev_emp = True
				#elif _tmp_strip.startswith("return ") and (not prev_emp):
				#	cache.append("")
				#	cache.append(tmp)
				#	prev_emp = False
				#else:
				cache.append(tmp)
				prev_emp = False
			else:
				if not prev_emp:
					cache.append(tmp.decode("utf-8"))
				prev_emp = True
	while not cache[-1]:
		del cache[-1]
	ens = "\n".encode("utf-8")
	with open(fname, "wb") as f:
		f.write("\n".join(cache).encode("utf-8"))
		f.write(ens)

def walk_path(ptws):

	for ptw in ptws:
		if isfile(ptw):
			if check_file_type(ptw):
				clean_tab(ptw)
		else:
			for root, dirs, files in walk(ptw):
				for execf in files:
					if check_file_type(execf):
						_execf = pjoin(root, execf)
						clean_tab(_execf)

if __name__ == "__main__":
	walk_path(sys.argv[1:])
