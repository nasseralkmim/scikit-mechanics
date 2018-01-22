import numpy as np
import time
from ..dirichlet import dirichlet
from ..neumann import neumann
from ..constructor import constructor
from ..postprocess.dof2node import dof2node


def solver(model, t=1):
    """Solver for the elastostatics problem

    Parameters
    ----------
    model : Build instance
        object containing all problem paramenters

   Return
    -------
    u : dict
        dictionary with node id and displacement

    """
    start = time.time()
    print('Starting statics solver at {:.3f}h '.format(t / 3600), end='')
    K, P = 0, 0
    for eid, [etype, *edata] in model.elements.items():
        element = constructor(eid, etype, model)
        k = element.stiffness_matrix(t)
        # pb = element.load_body_vector(model.body_force, t)
        # pe = element.load_strain_vector(t)
        K += k
        # P += pb + pe

    Pt = neumann(model)
    P = P + Pt
    Km, Pm = dirichlet(K, P, model)
    U = np.linalg.solve(Km, Pm)
    # add current dof displacement to model
    model.set_dof_displacement(U)
    u = dof2node(U, model)
    end = time.time()
    print('Solution completed in {:.3f}s!'.format(end - start))
    return u
