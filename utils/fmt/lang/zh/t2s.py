#encoding: utf-8

try:
	from opencc import OpenCC
except Exception as e:
	print(e)
	OpenCC = None
	from utils.func import identity_func

build_func = identity_func if OpenCC is None else (lambda task: OpenCC("%s.json" % task).convert)

t2s_func = build_func("t2s")
