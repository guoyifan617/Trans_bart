#encoding: utf-8

import sys

# cratio: number of "@@" ended tokens / number of all tokens
# bratio: number of bpe tokens / number of tokens before bpe processing
# sratio: number of tokens seperated by bpe / number of tokens before bpe processing
# num_rules_drop: choose from [1, 4], fewer data will be droped with larger value, none data would be droped if it was set to 4

from utils.fmt.base import sys_open

def handle(srcfs, tgtfs, cratio=0.8, bratio=5.0, sratio=0.8, num_rules_drop=1):

	def legal_mono(strin, cratio, bratio, sratio):
		ntokens = nchars = nsp = nrule = 0
		pbpe = False
		for tmpu in strin.split():
			if tmpu:
				if tmpu.endswith("@@"):
					nchars += 1
					if not pbpe:
						pbpe = True
						nsp += 1
				elif pbpe:
					pbpe = False
				ntokens += 1
		ntokens = float(ntokens)
		lorigin = float(len(strin.replace("@@ ", "").split()))
		if float(nchars) / ntokens > cratio:
			nrule += 1
		if ntokens / lorigin > bratio:
			nrule += 1
		if float(nsp) / lorigin > sratio:
			nrule += 1
		return nrule, ntokens, lorigin

	def legal(strins, cratio, bratio, sratio, num_rules_drop):

		ls, lens, lenso = legal_mono(strins, cratio, bratio, sratio)
		return ls < num_rules_drop

	ens = "\n".encode("utf-8")

	with sys_open(srcfs, "rb") as fs, sys_open(tgtfs, "wb") as fsw:
		total = keep = 0
		if num_rules_drop > 0:
			for ls in fs:
				ls = ls.strip()
				if ls:
					ls = ls.decode("utf-8")
					if (num_rules_drop > 3) or legal(ls, cratio, bratio, sratio, num_rules_drop):
						fsw.write(ls.encode("utf-8"))
						fsw.write(ens)
						keep += 1
					total += 1
		print("%d in %d data keeped with ratio %.2f" % (keep, total, float(keep) / float(total) * 100.0 if total > 0 else 0.0))

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]))
