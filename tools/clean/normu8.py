#encoding: utf-8

import sys

from utils.fmt.u8 import norm_u8_byte, uni_normer

def handle(srcf, rsf, uni_normer=uni_normer):

	ens="\n".encode("utf-8")
	with sys.stdin.buffer if srcf == "-" else open(srcf, "rb") as frd, sys.stdout.buffer if rsf == "-" else open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				fwrt.write(norm_u8_byte(tmp))
			fwrt.write(ens)

if __name__ == "__main__":
	handle(*sys.argv[1:])
