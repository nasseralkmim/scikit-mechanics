"""Constructs the correct element object

"""
from .elements.quad4 import Quad4
from .elements.quad4enr import Quad4Enr


def constructor(eid, etype, model):
    """Function that constructs the correct element

    """
    if etype == 3:
        if model.xfem is not None:
            if eid in model.xfem.enr_elements:
                return Quad4Enr(eid, model)
        else:
            return Quad4(eid, model)
    else:
        raise Exception('Element not implemented yet!')
