#encoding: utf-8

import sys

from utils.fmt.lang.zh.t2s import t2s_func
from utils.fmt.vocab.token import ldvocab_freq, save_vocab

def filter_func(x, vcb):

	_ = t2s_func(x)

	return (_ == x) or (not (_ in vcb))

def handle(srcf, rsf, vsize=65532):

	_vcb_freq = ldvocab_freq(srcf)[0]
	save_vocab({k: v for k, v in _vcb_freq.items() if filter_func(k, _vcb_freq)}, rsf, omit_vsize=vsize)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
