#encoding: utf-8

import sys

from utils.fmt.base import clean_liststr_lentok

def handle(srcfs, srcfm, srtsf, srtmf, srttf, tgtf):

	data = {}

	with open(srtsf, "rb") as fs, open(srtmf, "rb") as fm, open(srttf, "rb") as ft:
		for sl, ml, tl in zip(fs, fm, ft):
			_sl, _ml, _tl = sl.strip(), ml.strip(), tl.strip()
			if _tl:
				_sl, _ls = clean_liststr_lentok(_sl.decode("utf-8").split())
				_ml, _lm = clean_liststr_lentok(_ml.decode("utf-8").split())
				_tl, _lt = clean_liststr_lentok(_tl.decode("utf-8").split())
			data[(_sl, _ml)] = _tl

	ens = "\n".encode("utf-8")

	with open(srcfs, "rb") as fs, open(srcfm, "rb") as fm, open(tgtf, "wb") as ft:
		for line, linem in zip(fs, fm):
			tmp = line.strip()
			tmpm = linem.strip()
			tmp, _ = clean(tmp.decode("utf-8"))
			tmpm, _ = clean(tmpm.decode("utf-8"))
			tmp = data.get((tmp, tmpm), "")
			ft.write(tmp.encode("utf-8"))
			ft.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
