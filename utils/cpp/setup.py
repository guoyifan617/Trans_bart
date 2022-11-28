#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="movavg_cpp", ext_modules=[cpp_extension.CppExtension("movavg_cpp", ["movavg.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
