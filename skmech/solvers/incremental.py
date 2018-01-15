"""Solves the incremental problem

call this with:

    skmech.solvers.incremental()

"""
import numpy as np
import time
from ..dirichlet import dirichlet
from ..neumann import neumann
from ..constructor import constructor
from ..plasticity.stateupdatemises import state_update_mises
from ..plasticity.tangentmises import consistent_tangent_mises


def incremental(model, num_load_increments,
                max_iter_num=5, tol=1e-6):
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
    # TODO: Question about the initial load increment
    # does it starts at zero or at a very small number
    load_increment = np.linspace(1e-6, 1, num_load_increments)

    try:
        num_dof = model.num_dof
    except AttributeError:
        print('Model object does not have num_dof attribute')

    # initial displacement
    u = np.zeros(num_dof)

    # initial elastic strain for each element gauss point
    # TODO: fixed for 4 gauss point for now
    # TODO: maybe put that in the element class
    # TODO: Question about initial elastic strain
    eps_e_n = {eid: np.zeros((4, 3)) + 1e-6 for eid in model.elements.keys()}
    # initialize cummulatice plastic strain for each element gauss point
    eps_bar_p_n = {eid: np.zeros(4) for eid in model.elements.keys()}

    # external load vector
    # Only traction for now
    # TODO: external_load_vector function
    f_ext_bar = external_load_vector(model)

    f_int = internal_load_vector(model, u)
    # Loop over load increments
    for lmbda in load_increment:
        print(f'Load increment {lmbda}:', end=' ')
        f_ext = lmbda * f_ext_bar  # load vector for this pseudo time step

        # initial displacement increment for all dof
        du = np.zeros(num_dof)

        # Begin global Newton-Raphson
        for k in range(1, max_iter_num):
            # print(f'Load increment {lmbda} at {k} iteration')
            # initialize global vector and matrices
            f_int = np.zeros(num_dof)
            K_T = np.zeros((num_dof, num_dof))

            # Loop over elements
            for eid, [etype, *edata] in model.elements.items():
                # create element object
                element = constructor(eid, etype, model)
                # recover element nodal displacement increment,  shape (8,)
                dof = np.array(element.dof) - 1  # numpy starts at 0
                du_ele = du[dof]

                # material properties
                E, nu = element.E, element.nu
                # Hardening modulus
                # TODO: include this as a parameter of the material later
                H = 1e6
                sig_y0 = 1e5

                # initialize array for element internal force vector
                f_int_e = np.zeros(8)
                # initialize array for element consistent tangent matrix
                k_T_e = np.zeros((8, 8))

                # gauss point index -> only 4 gauss points for now
                gp_id = 0
                # loop over quadrature points
                for w, gp in zip(element.gauss.weights, element.gauss.points):
                    # build element strain-displacement matrix shape (3, 8)
                    N, dN_ei = element.shape_function(xez=gp)
                    dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)
                    B = element.gradient_operator(dN_xi)

                    # compute strain increment from
                    # current displacement increment, shape (3, )
                    deps = B @ du_ele

                    # elastic trial strain
                    eps_e_trial = eps_e_n[eid][gp_id, :] + deps

                    # trial cummulative plastic strain
                    eps_bar_p_trial = eps_bar_p_n[eid][gp_id]

                    # update internal variables for this gauss point
                    sig, eps_e, eps_bar_p, dgama, ep_flag = state_update_mises(
                        E, nu, H, sig_y0, eps_e_trial, eps_bar_p_trial)

                    # update elastic strain for this element for this gp
                    eps_e_n[eid][gp_id, :] = eps_e
                    # update cummulative plastic strain
                    eps_bar_p_n[eid][gp_id] = eps_bar_p
                    # update index for gauss point
                    gp_id += 1

                    # compute element internal force (gaussian quadrature)
                    f_int_e += B.T @ sig * (dJ * w)

                    # TODO: material properties from element, E, nu, H
                    # TODO: ep_flag comes from the state update? DONE
                    D = consistent_tangent_mises(
                        dgama, sig, E, nu, H, ep_flag)

                    # element consistent tanget matrix (gaussian quadrature)
                    k_T_e += B.T @ D @ B * (dJ * w)

                # Build global matrices outside the quadrature loop
                f_int[element.id_v] += f_int_e
                K_T[element.id_m] += k_T_e

            # compute global residual vector
            r = f_int - f_ext
            # apply boundary conditions modify mtrix and vectors
            K_T_m, r_m = dirichlet(K_T, r, model)
            # compute displacement increment on the NR k loop
            du = - np.linalg.solve(K_T_m, r_m)
            # update displacement with increment
            u += du

            # check convergence
            err = np.linalg.norm(r) / np.linalg.norm(f_ext)
            if err <= tol:
                # solution converged
                print(f'Converged with {k} iterations error {err:.3e}')
                # TODO: save internal variables to a file
                break
            else:
                continue

    end = time.time()
    print(f'Solution finished in {end - start:.3f}s')
    return None


def internal_load_vector(model, u):
    """Assemble internal load vector

    Note
    ----
    Reference Eq. 4.65 (1) Neto 2008

    Find the stress for a fiven displacement u, then multiply the stress for
    the strain-displacement matrix trasnpose and integrate it over domain.

    Procedure:
    1. Loop over elements
    2. loop over each gauss point
    3. compute the consistent tangent matrix using the constitutive equations
        3.1 Use elastic trail strain
        3.2 Use Von Mises model

    """
    return None


def external_load_vector(model):
    """Assemble external load vector

    Note
    ----
    Reference Eq. 4.68 Neto 2008
    Loop over elements and assemble load vector due tractions

    """
    # TODO: add body force later
    # only traction vector for now
    # not necessary to loop over elements
    Pt = neumann(model)
    return Pt


if __name__ == '__main__':
    import skmech

    class Mesh:
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
    displacement_bc = {12: (0, 0), 13: (None, 0)}
    model = skmech.Model(
        msh,
        material=material,
        traction=traction,
        displacement_bc=displacement_bc,
        num_quad_points=2)

    incremental(model, num_load_increments=20)
