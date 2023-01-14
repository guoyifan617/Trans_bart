#encoding: utf-8

from opencc import OpenCC

build_func = lambda task: OpenCC("%s.json" % task).convert

t2s_func = build_func("t2s")
