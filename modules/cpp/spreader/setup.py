#encoding: utf-8

from setuptools import setup
from torch.utils import cpp_extension

setup(name="spreader_cpp", ext_modules=[cpp_extension.CppExtension("spreader_cpp", ["modules/cpp/spreader/spreader.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
setup(name="spreader_nocx_cpp", ext_modules=[cpp_extension.CppExtension("spreader_nocx_cpp", ["modules/cpp/spreader/spreader_nocx.cpp"])], cmdclass={"build_ext": cpp_extension.BuildExtension})
