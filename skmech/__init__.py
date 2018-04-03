__author__ = 'Nasser'
__version__ = 0.2

from .model import Model
from .mesh import gmsh
from .mesh.mesh import Mesh
from .material import Material
from .solvers import statics
from .solvers import incremental
# from .postprocess import plotter
from .constructor import constructor
from .neumann import neumann
from .dirichlet import dirichlet
from . import postprocess
from . import xfem
from .meshplotlib.plot2d import plot
from .multiscale.micromodel import MicroModel
