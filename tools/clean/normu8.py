#encoding: utf-8

import sys
from html import unescape

def clean_line(istr):

	rs = []
	for c in istr:
		num = ord(c)
		if num == 12288:
			rs.append(" ")
		elif (num > 65280) and (num < 65375):
			rs.append(chr(num - 65248))
		elif not ((num < 32 and num != 9) or (num > 126 and num < 161) or (num > 8202 and num < 8206) or (num > 57343 and num < 63744) or (num > 64975 and num < 65008) or (num > 65519)):
			rs.append(c)

	return ''.join(rs)

def handle(srcf, rsf):

	ens="\n".encode("utf-8")
	with sys.stdin.buffer if srcf == "-" else open(srcf, "rb") as frd, sys.stdout.buffer if rsf == "-" else open(rsf, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clean_line(unescape(tmp.decode("utf-8"))).encode("utf-8")
				fwrt.write(tmp)
			fwrt.write(ens)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2])
