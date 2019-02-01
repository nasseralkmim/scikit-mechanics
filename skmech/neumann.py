"""Apply the Neumann boundary conditions"""
import numpy as np


def neumann(model):
    """Creates an equivalent nodal load with traction boundary condition

    """
    Pt = np.zeros(model.num_dof)
    if model.traction is not None:
        for t_location, t_vector in model.traction.items():
            physical_element = model.get_physical_element(t_location)
            if len(physical_element) == 0:
                raise Exception('Check if the physical element '
                                f'{physical_element} '
                                'was defined in gmsh')
            # loop over elements that are in the t_location physical entity
            for eid, [etype, *edata] in physical_element.items():
                # check if physical element is 2 node line
                if etype == 1:
                    n1, n2 = edata[-2], edata[-1]
                    dxyz = (np.asarray(model.mesh.nodes[n1]) -
                            np.asarray(model.mesh.nodes[n2]))
                    d = np.linalg.norm(dxyz)
                    pt = d * np.asarray(t_vector) / 2
                    dof1 = np.asarray(model.nodes_dof[n1]) - 1  # starts at 0
                    dof2 = np.asarray(model.nodes_dof[n2]) - 1  # starts at 0
                    Pt[dof1] += pt
                    Pt[dof2] += pt
                # if physical element is a node
                if etype == 15:
                    n = edata[-1]
                    dof = np.asarray(model.nodes_dof[n]) - 1  # starts at 0
                    pt = np.asarray(t_vector)
                    Pt[dof] += pt
    return Pt
