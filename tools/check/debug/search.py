#encoding: utf-8

import sys
from os import walk
from os.path import isfile, join as pjoin

def search_file(fname, cstr, detail=False):

	rs = [] if detail else False
	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.rstrip()
			if tmp:
				tmp = tmp.decode("utf-8")
				if tmp.find(cstr) >= 0:
					if detail:
						rs.append(tmp)
					else:
						rs = True
						break

	return rs

def walk_path(ptws, cstr, detail=True):

	for ptw in ptws:
		if isfile(ptw):
			rs = search_file(ptw, cstr, detail=detail)
			if rs:
				print(ptw)
				if detail:
					print("\n".join(rs))
		else:
			for root, dirs, files in walk(ptw):
				for pyf in files:
					if pyf.endswith(".py"):
						_pyf = pjoin(root, pyf)
						rs = search_file(_pyf, cstr, detail=detail)
						if rs:
							print(_pyf)
							if detail:
								print("\n".join(rs))

if __name__ == "__main__":
	walk_path(sys.argv[1:-1], sys.argv[-1])
