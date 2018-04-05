"""Solves the incremental problem"""
import numpy as np
import time
from ..neumann import neumann
from .increment import increment_step


def solver(model, time_step=.1, min_time_step=1e-3,
           max_num_iter=15, tol=1e-6,
           max_num_local_iter=100,
           element_out=None, node_out=None):
    """Performes the incremental solution of linearized virtual work equation

    Parameters
    ----------
    model : Model object
        object that contains the problem parameters: mesh, bc, etc
    time_step : float (0.1)
        pseudo time step to increment the analysis
    min_time_step : float (1e-3)
        minimum time step allowed when the step is divided when the number of
        iterations is greater than max_num_iteration

    Note
    ----
    Reference Box 4.2 (Neto 2008)

    Note
    ----
    Xfem and incremental analysis not tested.
    Using model.num_dof will change if using xfem.

    """
    start = time.time()
    print('Starting incremental solver')

    # initial displacement for t_0 (n=0)
    u = np.zeros(model.num_dof)

    int_var = initial_values(model)

    if model.micromodel is not None:
        # add microscale variables into int_var dict
        int_var = model.micromodel.set_internal_var(
            int_var)

    # external load vector, only traction for now
    f_ext_bar = external_load_vector(model)

    increment, lmbda = 0, 0
    # Loop over load increments
    while lmbda <= 1 + tol:
        print('--------------------------------------')
        print(f'Load factor {lmbda:.4f} increment {increment}')
        print('--------------------------------------')

        # break after all imposed displacement load steps (for displ control)
        if model.imposed_displ is not None:
            if increment >= len(model.imposed_displ):
                break

        f_ext = lmbda * f_ext_bar

        # perform an increment and update displacement and internal variables
        u, int_var = increment_step(model, u, int_var, f_ext,
                                    max_num_iter, max_num_local_iter,
                                    increment, tol, lmbda, start,
                                    element_out, node_out)

        # add to time step
        lmbda = lmbda + time_step
        increment += 1

    end = time.time()
    print(f'Solution finished in {end - start:.3f}s')
    return None


def external_load_vector(model):
    """Assemble external load vector

    Note
    ----
    Reference Eq. 4.68 (Neto 2008)

    """
    # TODO: add body force later
    # only traction vector for now
    Pt = neumann(model)
    return Pt


def initial_values(model):
    """Initialize internal variables dic

    Note
    ----
    The elastic strain for plane strain case has to consider the zz=33
    component, because even though the total strain eps_33 = 0, the elastic
    and platic parts are not, See (p. 761 Neto 2008)

    """
    # initial elastic strain for each element gauss point
    # TODO: fixed for 4 gauss point for now
    # TODO: maybe put that in the element class
    # TODO: Question about initial elastic strain
    # initialize cummulatice plastic strain for each element gauss point
    # num_quad_points is a dictionary with {eid, num_quad_points}
    # num_quad_poins for each dimension, multiply by 2 for plane problems
    eps_e_n = {(eid, gp): np.zeros(4) for eid in model.elements.keys()
               for gp in range(model.num_quad_points[eid] * 2)}
    eps_p_n = {(eid, gp): np.zeros(4) for eid in model.elements.keys()
               for gp in range(model.num_quad_points[eid] * 2)}
    eps_bar_p_n = {(eid, gp): 0 for eid in model.elements.keys()
                   for gp in range(model.num_quad_points[eid] * 2)}
    # initialize dict to store incremental plastic multiplier
    # used to compute the consistent tangent matrix
    dgamma_n = {(eid, gp): 0 for eid in model.elements.keys()
                for gp in range(model.num_quad_points[eid] * 2)}
    # initialize displacement to compute internal force vector
    # at firt iteration of each step
    # sig_n = {(eid, gp): np.zeros(3) for eid in model.elements.keys()
    # for gp in range(model.num_quad_points[eid] * 2)}
    int_var = {'eps_e': eps_e_n,
               'eps': eps_e_n,
               'eps_p': eps_p_n,
               'eps_bar_p': eps_bar_p_n,
               'dgamma': dgamma_n}
    return int_var

