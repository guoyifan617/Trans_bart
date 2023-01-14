#encoding: utf-8

import sys

from utils.fmt.lang.zh.t2s import t2s_func
from utils.fmt.vocab.char import ldvocab_freq, save_vocab

def filter_func(x, vcb):

	_ = t2s_func(x)

	return (_ == x) or (not (_ in vcb))

def handle(srcf, rsf, vsize=65532):

	_vcb_freq = ldvocab_freq(srcf, omit_vsize=vsize)[0]
	save_vocab({k: v for k, v in _vcb_freq.items() if filter_func(k, _vcb_freq)}, rsf)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], int(sys.argv[3]))
