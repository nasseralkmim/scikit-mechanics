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
            if model.micromodel is not None:
                # if mulsticale analys set any material parameter
                # in multiscale analysis there is no need for specifiying
                # material in the macro model

                class Material:
                    pass
                phy_surf = model.elements[eid][2]
                model.material = Material
                model.material.case = 'strain'
                model.material.E, model.material.nu = {}, {}
                model.material.H, model.material.sig_y0 = {}, {}
                model.material.E[phy_surf] = 1
                model.material.nu[phy_surf] = 0
                model.material.H[phy_surf] = 0
                model.material.sig_y0[phy_surf] = 0
                return Quad4(eid, model)
            else:
                # regular analysis
                return Quad4(eid, model)
    else:
        raise Exception('Element not implemented yet!')
