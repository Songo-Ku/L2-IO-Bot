import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("Agency",  ["Agency.py"]),
    Extension("Agent",  ["Agent.pyw"]),
    Extension("Vision",  ["Vision.py"]),
    Extension("Satellite",  ["Satellite.pyw"]),
    Extension("Constants",  ["Constants.py"])

#   ... all your modules that need be compiled ...

]

setup(
    name = 'My Program Name',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules
)