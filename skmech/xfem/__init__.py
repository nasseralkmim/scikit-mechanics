"""import process and zerolevelset so I can

    >>> from elastopy import xfem

then this will run xfem/__init__.py, which will import process and zerolevelset 
that will be available with:

     >>> xfem.process.func()

"""
from . import postprocess
from . import zerolevelset
