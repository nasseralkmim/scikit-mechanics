"""Solves the incremental problem"""
import numpy as np
import time
from ..dirichlet import imposed_displacement
from ..neumann import neumann
from .localization import localization
from ..postprocess.saveoutput import save_output
from .partitioned import solve_partitioned


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

    """
    start = time.time()
    print('Starting incremental solver')

    # NN why would model not have num_dof???
    try:
        num_dof = model.num_dof
    except AttributeError:
        raise Exception('Model object does not have num_dof attribute')

    # initial displacement for t_0 (n=0)
    u = np.zeros(num_dof)

    eps_e_n, eps_p_n, eps_bar_p_n, dgamma_n = initial_values(model)

    # external load vector
    # Only traction for now
    f_ext_bar = external_load_vector(model)

    increment, lmbda = 0, 0
    # Loop over load increments
    while lmbda <= 1 + tol:
        print('--------------------------------------')
        print(f'Load factor {lmbda:.4f} increment {increment}')
        print('--------------------------------------')

        # break after all imposed displacement load steps
        if model.imposed_displ is not None:
            if increment >= len(model.imposed_displ):
                break

        # initial displacement increment for each load step
        Delta_u = np.zeros(num_dof)
        f_ext = lmbda * f_ext_bar
        # Step (2), (3)
        f_int, K_T, int_var = localization(model, Delta_u,
                                           eps_e_n,
                                           eps_p_n,
                                           eps_bar_p_n,
                                           dgamma_n,
                                           max_num_local_iter)
        # Begin global Newton procedures
        for k in range(0, max_num_iter + 1):
            # if more than 6 iterations, add half of the interval
            if k >= max_num_iter:
                lmbda = lmbda - time_step + time_step / 2
                if time_step < min_time_step:
                    time_step = min_time_step
                else:
                    time_step = time_step / 2
                # break out of Newton loop
                break

            # Step (4) Assemble global and solve for correction
            newton_correction, f_ext = solve_partitioned(
                model, K_T, f_int, f_ext, increment, k)
            # Step (5) Update solutions
            Delta_u += newton_correction
            u += newton_correction
            # Step (6) (7) (8)
            # build internal load vector and solve local constitutive equation
            f_int, K_T, int_var = localization(model, Delta_u,
                                               eps_e_n,
                                               eps_p_n,
                                               eps_bar_p_n,
                                               dgamma_n,
                                               max_num_local_iter)
            # new residual
            r_updt = f_int - f_ext
            # compute residual norm to check equilibrium
            r_norm = np.linalg.norm(r_updt)

            print(f'Iteration {k + 1} residual norm {r_norm:.1e}')

            if r_norm <= tol:
                # solution converged +1 because it started in 0
                print(f'Converged with {k + 1} iterations '
                      f'residual norm {r_norm:.1e}')

                # add to time step
                lmbda = lmbda + time_step
                increment += 1

                eps_e_n, eps_p_n, eps_bar_p_n, dgamma_n = update_int_var(
                    int_var)
                save_output(model, u, int_var, increment, start, lmbda,
                            element_out, node_out)
                break
            else:
                # did't converge, continue to next global iteration
                continue
        else:
            raise Exception(f'Solution did not converge at time step '
                            f'{increment + 1} after {k} iterations')
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
    return eps_e_n, eps_p_n, eps_bar_p_n, dgamma_n


def update_int_var(int_var):
    """Update internal variables with iteration values"""
    # TODO: update interal variables converged DONE
    # update elastic strain for this element for this gp
    eps_e_n = int_var['eps_e']
    # update plastic strain
    eps_p_n = int_var['eps_p']
    # update cummulative plastic strain
    eps_bar_p_n = int_var['eps_bar_p']
    # update incremental plastic multiplier
    dgamma_n = int_var['dgamma']
    # update stress, not required
    # sig_n = int_var['sig']
    return eps_e_n, eps_p_n, eps_bar_p_n, dgamma_n


if __name__ == '__main__':
    import skmech

    class Mesh():
        pass

    # 4 element with offset center node
    msh = Mesh()
    msh.nodes = {
        1: [0, 0, 0],
        2: [1, 0, 0],
        3: [1, 1, 0],
        4: [0, 1, 0],
        5: [.5, 0, 0],
        6: [1, .5, 0],
        7: [.5, 1, 0],
        8: [0, .5, 0],
        9: [.4, .6]
    }
    msh.elements = {
        1: [15, 2, 12, 1, 1],
        2: [15, 2, 13, 2, 2],
        3: [1, 2, 7, 2, 2, 6],
        4: [1, 2, 7, 2, 6, 3],
        7: [1, 2, 5, 4, 4, 8],
        8: [1, 2, 5, 4, 8, 1],
        9: [3, 2, 11, 10, 1, 5, 9, 8],
        10: [3, 2, 11, 10, 5, 2, 6, 9],
        11: [3, 2, 11, 10, 9, 6, 3, 7],
        12: [3, 2, 11, 10, 8, 9, 7, 4]
    }
    material = skmech.Material(E={11: 10000}, nu={11: 0.3})
    traction = {5: (-1, 0), 7: (1, 0)}
    displacement_bc = {12: (0, 0)}

    model = skmech.Model(
        msh,
        material=material,
        traction=traction,
        imposed_displ=imposed_displacement,
        displacement_bc=displacement_bc,
        num_quad_points=2)
