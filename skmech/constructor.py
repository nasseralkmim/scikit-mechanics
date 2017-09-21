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
                # catch non enriched elements
                # TODO: element material for non enriched element
                # update element material for matrix or reinforcement
                phy_surf = model.elements[eid][2]
                if eid in model.xfem.element_material['reinforcement']:
                    model.material.E[phy_surf] = model.xfem.material.E[-1]
                    model.material.nu[phy_surf] = model.xfem.material.nu[-1]
                elif eid in model.xfem.element_material['matrix']:
                    model.material.E[phy_surf] = model.xfem.material.E[1]
                    model.material.nu[phy_surf] = model.xfem.material.nu[1]
                return Quad4(eid, model)
        else:
            return Quad4(eid, model)
    else:
        raise Exception('Element not implemented yet!')
