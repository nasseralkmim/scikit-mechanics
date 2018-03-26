"""Solves the incremental  problem"""
import numpy as np
import time
from ..dirichlet import dirichlet
from ..dirichlet import imposed_displacement
from ..neumann import neumann
from ..constructor import constructor
from ..plasticity.stateupdatemises import state_update_mises
from ..plasticity.tangentmises import consistent_tangent_mises
from ..postprocess.dof2node import dof2node
from ..meshplotlib.gmshio.gmshio import write_field


def solver(model, load_increment=None,
           num_load_increments=10,
           max_num_iter=5, tol=1e-6,
           max_num_local_iter=100):
    """Performes the incremental solution of linearized virtual work equation

    Parameters
    ----------
    model : Model object
        object that contains the problem parameters: mesh, bc, etc

    Note
    ----
    Reference Box 4.2 Neto 2008

    """
    start = time.time()
    print('Starting incremental solver')

    # load increment
    # TODO: Question about the initial load increment DONE
    # does it starts at zero or at a very small number -> at 0
    # See feap examples manual
    if load_increment is None:
        load_increment = np.linspace(0, 1, num_load_increments)

    try:
        num_dof = model.num_dof
    except AttributeError:
        raise Exception('Model object does not have num_dof attribute')

    # numpy array to store solution for all dof
    displ = np.zeros((num_dof, len(load_increment)))

    # initial displacement for t_0 (n=0)
    u = np.zeros(num_dof)

    # initial elastic strain for each element gauss point
    # TODO: fixed for 4 gauss point for now
    # TODO: maybe put that in the element class
    # TODO: Question about initial elastic strain
    # initialize cummulatice plastic strain for each element gauss point
    # num_quad_points is a dictionary with {eid, num_quad_points}
    # num_quad_poins for each dimension, multiply by 2 for plane problems
    eps_e_n = {(eid, gp): np.zeros(3) for eid in model.elements.keys()
               for gp in range(model.num_quad_points[eid] * 2)}
    eps_bar_p_n = {(eid, gp): 0 for eid in model.elements.keys()
                   for gp in range(model.num_quad_points[eid] * 2)}
    # initialize dict to store incremental plastic multiplier
    # used to compute the consistent tangent matrix
    dgamma_n = {(eid, gp): 0 for eid in model.elements.keys()
                for gp in range(model.num_quad_points[eid] * 2)}
    # initialize displacement to compute internal force vector
    # at firt iteration of each step
    sig_n = {(eid, gp): np.zeros(3) for eid in model.elements.keys()
             for gp in range(model.num_quad_points[eid] * 2)}

    # external load vector
    # Only traction for now
    # TODO: external_load_vector function DONE for traction only
    f_ext_bar = external_load_vector(model)

    # if model.imposed_displ is not None:
    #     free_dof, restrained_dof = get_free_restrained_dof(model)
    #     # set up incidence for free dofs
    #     ff = np.ix_(free_dof, free_dof)
    #     fp = np.ix_(free_dof, restrained_dof)

    # initial internal load vector
    f_int = np.zeros(num_dof)

    old_lmbda = 0
    # Loop over load increments
    for incr_id, lmbda in enumerate(load_increment):
        print('--------------------------------------')
        print(f'Load increment {lmbda:.2f}:')
        print('--------------------------------------')

        # load vector for this pseudo time step
        f_ext = lmbda * f_ext_bar

        # Initial parameters for N-R
        # Initial displacement increment for all dof
        du = np.zeros(num_dof)

        # initial stress for this time step, use the converged
        # value from previous step
        sig = sig_n

        # Begin global Newton-Raphson
        for k in range(0, max_num_iter):
            # build first tangent stiffness matrix
            # K_T(sig_k, internal_variables_n)
            # later it will be built on the local solver
            if k == 0:
                # build consistent tangent matrix using updated sig
                # ep_flag = False considers first iteration elastic always
                K_T = build_tangent_stiffness(model, num_dof, sig, dgamma_n,
                                              ep_flag=False)

                # apply boundary condition for displacement control case
                # apply at every time step only in the first iteration
                if model.imposed_displ is not None:
                    # Make a copy of the imposed displacement
                    imposed_displ = dict(model.imposed_displ)
                    for line, (d1, d2) in imposed_displ.items():
                        if d1 is not None:
                            d1 /= len(load_increment) * np.sign(lmbda -
                                                                old_lmbda)
                        if d2 is not None:
                            d2 /= len(load_increment) * np.sign(lmbda -
                                                                old_lmbda)
                        # update dictionary with this load factor
                        imposed_displ[line] = (d1, d2)
                    # update model displacement bc in order to enforce this
                    # load step displacement
                    model.displacement_bc.update(imposed_displ)
                    old_lmbda = lmbda

            # remove boundary conditions from model.displacement_bc list
            # so they don't add every iteration, only in the begining of
            # time step
            if k == 1:
                if model.imposed_displ is not None:
                    new_bc = {}
                    for line, (d1, d2) in imposed_displ.items():
                        model.displacement_bc.pop(line)
                        new_bc[line] = (0, 0)
                    # add 0, 0 displacement so the imposed displacement does
                    # not change
                    model.displacement_bc.update(new_bc)

            # compute global residual vector
            r = f_ext - f_int

            # apply boundary conditions modify mtrix and vectors
            K_T_m, r_m = dirichlet(K_T, r, model)
            # compute the N-R correction (delta u)
            newton_correction = np.linalg.solve(K_T_m, r_m)

            # displacement increment on the NR k loop, du starts at 0
            # for each load step
            du += newton_correction
            # update displacement with increment
            u += newton_correction
            # build internal load vector and solve local constitutive equation
            f_int, K_T, int_var = local_solver(model, num_dof, du,
                                               eps_e_n,
                                               eps_bar_p_n,
                                               dgamma_n,
                                               max_num_local_iter)
            # new residual
            r_updt = f_int - f_ext

            # get reference values to check convergence
            if k == 0:
                r_ref = r_updt
                newton_correction_ref = newton_correction

            convergence, error, error_type = convergence_tests(
                newton_correction, r_updt,
                newton_correction_ref, r_ref, tol)

            print(f'Iteration {k + 1} error {error:.1e}/type: {error_type}')

            if convergence is True:
                # solution converged +1 because it started in 0
                print(f'Converged with {k + 1} iterations error {error:.1e}/'
                      f'type: {error_type}')

                # TODO: store variable in an array DONE
                displ[:, incr_id] = u

                # TODO: update interal variables converged DONE
                # update elastic strain for this element for this gp
                eps_e_n = int_var['eps_e']
                # update cummulative plastic strain
                eps_bar_p_n = int_var['eps_bar_p']
                # update incremental plastic multiplier
                dgamma_n = int_var['dgamma']
                # update stress
                sig_n = int_var['sig']

                # TODO: save internal variables to a file DONE
                displ_dic = dof2node(u, model)
                write_field(displ_dic, model.mesh.name,
                            'Displacement', 2, lmbda, incr_id, start)

                # element average of cummulative plastic strain
                eps_bar_p_avg = {eid: eps_bar_p_n[(eid, gp)]
                                 for eid in model.elements.keys()
                                 for gp in range(4)}
                write_field(eps_bar_p_avg, model.mesh.name,
                            'Cummulative plastic strain', 1,
                            lmbda, incr_id, start, datatype='Element')

                # sig_ele {eid: [sig_x, sig_y, sig_xy]}
                sig_ele = int_var['sig_ele']
                sig_x = {eid: sig_ele[eid][0] for eid in model.elements.keys()}
                write_field(sig_x, model.mesh.name, 'Sigma x', 1,
                            lmbda, incr_id, start, datatype='Element')

                break
            else:
                # did't converge, continue to next global iteration
                continue
        else:
            raise Exception(f'Solution did not converge at time step '
                            f'{incr_id + 1} after {k} iterations with error '
                            f'{err:.1e}')
    end = time.time()
    print(f'Solution finished in {end - start:.3f}s')
    return displ, load_increment


def solve_partitioned(model, K_T, residual):
    """Solve partitioned system

    Obtain Newton correction for free degree's of freedom and obtain residual
    for restrained degree's of freedom

    [ Kff Kfr ] [ delta_u_f ]     [ residual_f ]
    [ Krf Krr ] [ delta_u_r ] = - [ residual_r ]

    Returns
    -------
    delta_u, r
        newton correction and residual orderer as [free, restrained] dof

    Note
    ----
    Considering homogeneous Dirichlet boundary conditions, zero displacement
    on the restrained dofs

    """
    # incidences of free and restrained dofs
    f, r = model.id_f, model.id_r
    ff = np.ix_(f, f)
    rf = np.ix_(r, f)

    # updtade vector with all dofs, the restrained dofs correction is zero
    delta_u = np.zeros(model.num_dof)
    # solve for free dofs
    delta_u[f] = np.linalg.solve(K_T[ff], - residual[f])

    # update residual vector with the restrained part
    residual[model.id_r] = - K_T[rf] @ delta_u[f]

    return delta_u, residual


def external_load_vector(model):
    """Assemble external load vector

    Note
    ----
    Reference Eq. 4.68 Neto 2008

    """
    # TODO: add body force later
    # only traction vector for now
    Pt = neumann(model)
    return Pt


def initial_values(model):
    """Initialize internal variables dic"""
    # initial elastic strain for each element gauss point
    # TODO: fixed for 4 gauss point for now
    # TODO: maybe put that in the element class
    # TODO: Question about initial elastic strain
    # initialize cummulatice plastic strain for each element gauss point
    # num_quad_points is a dictionary with {eid, num_quad_points}
    # num_quad_poins for each dimension, multiply by 2 for plane problems
    eps_e_n = {(eid, gp): np.zeros(3) for eid in model.elements.keys()
               for gp in range(model.num_quad_points[eid] * 2)}
    eps_bar_p_n = {(eid, gp): 0 for eid in model.elements.keys()
                   for gp in range(model.num_quad_points[eid] * 2)}
    # initialize dict to store incremental plastic multiplier
    # used to compute the consistent tangent matrix
    dgamma_n = {(eid, gp): 0 for eid in model.elements.keys()
                for gp in range(model.num_quad_points[eid] * 2)}
    # initialize displacement to compute internal force vector
    # at firt iteration of each step
    sig_n = {(eid, gp): np.zeros(3) for eid in model.elements.keys()
             for gp in range(model.num_quad_points[eid] * 2)}
    return eps_e_n, eps_bar_p_n, dgamma_n, sig_n


def update_int_var(int_var):
    """Update internal variables with iteration values"""
    # TODO: update interal variables converged DONE
    # update elastic strain for this element for this gp
    eps_e_n = int_var['eps_e']
    # update cummulative plastic strain
    eps_bar_p_n = int_var['eps_bar_p']
    # update incremental plastic multiplier
    dgamma_n = int_var['dgamma']
    # update stress
    sig_n = int_var['sig']
    return eps_e_n, eps_bar_p_n, dgamma_n, sig_n


def save_output(model, u, int_var, increment, start, lmbda,
                element_out, node_out):
    """Save output to .msh file"""
    # TODO: save internal variables to a file DONE
    displ_dic = dof2node(u, model)
    write_field(displ_dic, model.mesh.name,
                'Displacement', 2, lmbda, increment, start)
    if node_out is not None:
        write_output(lmbda, displ_dic[node_out],
                     f'displ_node{node_out}', start)
    # smoothed (average) extrapolated stresses to nodes
    sig_dic = stress_recovery_smoothed(model, u)
    sig_x_dic = {nid: sx for nid, [sx, _, _] in sig_dic.items()}
    sig_y_dic = {nid: sy for nid, [_, sy, _] in sig_dic.items()}
    sig_xy_dic = {nid: txy for nid, [_, _, txy] in sig_dic.items()}
    write_field(sig_x_dic, model.mesh.name,
                'Sigma x', 1, lmbda, increment, start)
    write_field(sig_y_dic, model.mesh.name,
                'Sigma y', 1, lmbda, increment, start)
    write_field(sig_xy_dic, model.mesh.name,
                'Sigma xy', 1, lmbda, increment, start)

    # element average of cummulative plastic strain
    eps_bar_p_avg = {eid: int_var['eps_bar_p'][(eid, gp)]
                     for eid in model.elements.keys()
                     for gp in range(4)}
    write_field(eps_bar_p_avg, model.mesh.name,
                'Cummulative plastic strain element average', 1,
                lmbda, increment, start, datatype='Element')
    if element_out is not None:
        write_output(lmbda, int_var['eps_bar_p'][(element_out, 0)],
                     f'peeq_gp1_ele{element_out}', start)
        write_output(lmbda, int_var['eps_bar_p'][(element_out, 1)],
                     f'peeq_gp2_ele{element_out}', start)
        write_output(lmbda, int_var['eps_bar_p'][(element_out, 2)],
                     f'peeq_gp3_ele{element_out}', start)
        write_output(lmbda, int_var['eps_bar_p'][(element_out, 3)],
                     f'peeq_gp4_ele{element_out}', start)

        write_output(lmbda, int_var['sig'][(element_out, 0)],
                     f'sig_gp1_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 1)],
                     f'sig_gp2_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 2)],
                     f'sig_gp3_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 3)],
                     f'sig_gp4_ele{element_out}', start)

        write_output(lmbda, int_var['eps'][(element_out, 0)],
                     f'eps_gp1_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 1)],
                     f'eps_gp2_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 2)],
                     f'eps_gp3_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 3)],
                     f'eps_gp4_ele{element_out}', start)
    return None


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
    imposed_displacement = {13: (10, None)}
    model = skmech.Model(
        msh,
        material=material,
        traction=traction,
        imposed_displ=imposed_displacement,
        displacement_bc=displacement_bc,
        num_quad_points=2)

    print(get_free_dof(model))


