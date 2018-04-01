"""Perform the localization procedure of FEM

Localization means to loop over elements, build element matrices and assemble
global

"""
import numpy as np
from ..constructor import constructor
from ..plasticity.stateupdatemises import state_update_mises as suvm
from ..plasticity.tangentmises import consistent_tangent_mises


def localization(model, Delta_u, eps_e_n, eps_p_n, eps_bar_p_n, dgamma_n,
                 max_num_local_iter):
    """Localization of fem procedure

    Parameters
    ----------
    model : Model object
    num_dof : number of degree's of freedom
    Delta_u : displacement increment
    eps_e_n : dict {(eid, gp_id): ndarray shape (4)}
        Stores the elastic strain at previous step for each element and each
        gauss point. This value is updated every time this function is called.
        Note that it has 4 components, the elastic strain component, eps_e_33,
        is not zero
    eps_p_n : dict {(eid, gp_id): ndarray shape (4)}
        Note that it has 4 components, the plastic strain component, eps_e_33,
        is not zero, even though the total strain eps_33 = 0 for plane strain.
    eps_bar_p_n : dict {(eid, gp_id): float}
        Stores the accumulated plastic strain at previous step for each element
        and each gauss point (gp). This value is updated every time this
        function is called
    dgamma_n : dict {(eid, gp_id): float}

    Returns
    -------
    f_int : ndarray shape (num_dof)
    K_T : ndarray shape (num_dof, num_dof)
    ep_flag : str
    int_var : dict
        dictionary of interal state variables for each gauss point for
        each element

    Note
    ----
    Reference Eq. 4.65 (1) Neto 2008

    Find the stress for a fiven displacement u, then multiply the stress for
    the strain-displacement matrix trasnpose and integrate it over domain.

    Procedure:
    1. Loop over elements
    2. Loop over each gauss point
        2.1 Compute strain increment from displacement increment
        2.2 Compute elastic trial strain
        2.3 Update state variables (stress, elastic strain, plastic multiplier,
                                    accumulated plastic strain)
        2.4 Compute internal element force vector
        2.5 Compute element tangent stiffness matrix
    3. Assemble global internal force vector and tangent stiffness matrix

    """
    num_dof = model.num_dof
    # initialize global vector and matrices
    f_int = np.zeros(num_dof)
    K_T = np.zeros((num_dof, num_dof))

    # dictionary with local variables
    # new every local N-R iteration
    # use to save converged value
    int_var = {'eps_e': {}, 'eps': {}, 'eps_bar_p': {},
               'dgamma': {}, 'sig': {}, 'eps_p': {}, 'q': {}}

    # Loop over elements
    for eid, [etype, *edata] in model.elements.items():
        # create element object
        element = constructor(eid, etype, model)
        # recover element nodal displacement increment,  shape (8,)
        dof = np.array(element.dof) - 1  # numpy starts at 0
        Delta_u_ele = Delta_u[dof]

        # material properties
        E, nu = element.E, element.nu
        # Hardening modulus and yield stress
        # TODO: include this as a parameter of the material later DONE
        try:
            H = model.material.H[element.physical_surf]
            sig_y0 = model.material.sig_y0[element.physical_surf]
        except (AttributeError, KeyError) as err:
            raise Exception('Missing material property H and sig_y0 in'
                            'the material object')

        # initialize array for element internal force vector and tangent
        f_int_e = np.zeros(8)
        k_T_e = np.zeros((8, 8))

        # loop over quadrature points
        for gp_id, [w, gp] in enumerate(zip(element.gauss.weights,
                                            element.gauss.points)):
            # build element strain-displacement matrix shape (3, 8)
            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)
            B = element.gradient_operator(dN_xi)

            # compute strain increment from
            # current displacement increment, shape (3, )
            Delta_eps = B @ Delta_u_ele
            Delta_eps = np.append(Delta_eps, 0)  # Delta_eps_zz = 0

            # elastic trial strain
            # use the previous value stored for this element and this gp
            eps_e_trial = eps_e_n[(eid, gp_id)] + Delta_eps

            # trial accumulated plastic strain
            # this is only updated when converged
            eps_bar_p_trial = eps_bar_p_n[(eid, gp_id)]

            # plastic strain trial is from previous load step
            eps_p_trial = eps_p_n[(eid, gp_id)]

            # update internal variables for this gauss point
            sig, eps_e, eps_p, eps_bar_p, dgamma, q, ep_flag = suvm(
                E, nu, H, sig_y0, eps_e_trial, eps_bar_p_trial, eps_p_trial,
                max_num_local_iter, model.material.case)
            int_var = storage_int_var(int_var, eid, gp_id, eps_e, eps_p, sig,
                                      eps_bar_p, q, dgamma, element)

            # compute element internal force (gaussian quadrature)
            # sig[:3] ignore the 33 component here
            f_int_e += B.T @ sig[:3] * (dJ * w * element.thickness)

            # TODO: material properties from element, E, nu, H DONE
            # TODO: ep_flag comes from the state update? DONE
            # use dgama from previous global iteration
            D = consistent_tangent_mises(
                dgamma_n[(eid, gp_id)], sig, E, nu, H, ep_flag,
                model.material.case)
            # print(D / 1e9, 'GPa')
            # element consistent tanget matrix (gaussian quadrature)
            k_T_e += B.T @ D @ B * (dJ * w * element.thickness)

        # Build global matrices outside the quadrature loop
        # += because elements can share same dof
        f_int[element.id_v] += f_int_e
        K_T[element.id_m] += k_T_e

    return f_int, K_T, int_var


def storage_int_var(int_var, eid, gp_id, eps_e, eps_p, sig,
                    eps_bar_p, q, dgamma, element):
    """Storage internal variables

    Parameters
    ----------

    Return
    ------
    int_var : dict
        dictionary of interal state variables for each gauss point for
        each element

    """
    # TODO Only update when converged! outside this function! DONE
    # save solution of constitutive equation -> internal variables
    # int_var is a dictionary with the internal variables
    # each interal variable is a dictionary with a tuple key
    # the tuple (eid, gp_id) for each element and each gauss point
    int_var['eps_e'][(eid, gp_id)] = eps_e
    int_var['eps_p'][(eid, gp_id)] = eps_p
    int_var['eps'][(eid, gp_id)] = eps_e + eps_p
    int_var['eps_bar_p'][(eid, gp_id)] = eps_bar_p
    int_var['dgamma'][(eid, gp_id)] = dgamma
    int_var['sig'][(eid, gp_id)] = sig
    int_var['q'][(eid, gp_id)] = q

    return int_var
