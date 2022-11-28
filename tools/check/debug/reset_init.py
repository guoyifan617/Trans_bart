#encoding: utf-8

import sys
from os import walk
from os.path import join as pjoin

def walk_path(ptw):

	ttw = "#encoding: utf-8\n".encode("utf-8")
	for root, dirs, files in walk(ptw):
		for pyf in files:
			if pyf == "__init__.py":
				_pyf = pjoin(root, pyf)
				with open(_pyf, "wb") as fwrt:
					fwrt.write(ttw)

if __name__ == "__main__":
	walk_path(sys.argv[1])
