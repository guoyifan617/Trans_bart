#encoding: utf-8

import sys

from utils.fmt.base import sys_open
from utils.fmt.doc.base import get_char_ratio

def handle(srcfs):

	def getfratio(fname):

		with sys_open(fname, "rb") as fs:

			maxratioc = 0.0
			maxratiob = 0.0
			maxratios = 0.0

			meanratioc = 0.0
			meanratiob = 0.0
			meanratios = 0.0

			ndata = 0
			cache = []
			for line in fs:
				line = line.strip()
				if line:
					cache.append(line.decode("utf-8"))
				else:
					if cache:
						ratioc, ratiob, ratios = get_char_ratio(cache)
						if ratioc > maxratioc:
							maxratioc = ratioc
						meanratioc += ratioc
						if ratiob > maxratiob:
							maxratiob = ratiob
						meanratiob += ratiob
						if ratios > maxratios:
							maxratios = ratios
						meanratios += ratios
						ndata += 1
						cache = []
			if cache:
				ratioc, ratiob, ratios = get_char_ratio(cache)
				if ratioc > maxratioc:
					maxratioc = ratioc
				meanratioc += ratioc
				if ratiob > maxratiob:
					maxratiob = ratiob
				meanratiob += ratiob
				if ratios > maxratios:
					maxratios = ratios
				meanratios += ratios
				ndata += 1
				cache = []

		ndata = float(ndata)
		meanratioc /= ndata
		meanratiob /= ndata
		meanratios /= ndata

		return maxratioc, meanratioc, maxratiob, meanratiob, maxratios, meanratios

	mrsc, _rsc, mrsb, _rsb, mrss, _rss = getfratio(srcfs)

	print("Max/mean/adv char ratio of source data: %.3f / %.3f / %.3f\nMax/mean/adv bpe ratio of source data: %.3f / %.3f / %.3f\nMax/mean/adv seperated ratio of source data: %.3f / %.3f / %.3f" % (mrsc, _rsc, min(mrsc, _rsc * 2.5) + 0.001, mrsb, _rsb, min(mrsb, _rsb * 2.5) + 0.001, mrss, _rss, min(mrss, _rss * 2.5) + 0.001,))

if __name__ == "__main__":
	handle(sys.argv[1])
