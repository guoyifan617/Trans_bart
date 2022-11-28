#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="mulang_eff_ppff_cpp", ext_modules=[cpp_extension.CppExtension("mulang_eff_ppff_cpp", ["modules/cpp/mulang/eff/pffn/pff.cpp", "modules/cpp/mulang/eff/pffn/pff_func.cpp", "modules/cpp/base/ffn/pff_func.cpp", "modules/cpp/act/act_func.cpp"], extra_compile_args=["-fopenmp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
