"""Perform the localization procedure of FEM multiscale

Localization means to loop over elements, build element matrices and assemble
global

TODO: instead of using suvm and ctvm use a microscale solver
TODO: maybe just add an if to the solve/localization file, but try here first
    Actually is a little different because it need to update the displacement
    fluctuation

"""
import numpy as np
from ..constructor import constructor
from ..plasticity.stateupdatemises import state_update_mises as suvm
from ..plasticity.tangentmises import consistent_tangent_mises


def micro_localization(model, Delta_u_fluct, int_var, Delta_eps_macro,
                       Meid, Mgpid,
                       max_num_local_iter):
    """Localization of fem procedure for micro scale

    Parameters
    ----------
    model : Model object
    Delta_u_fluct : ndarray
        displacement increment which is the sum of the corrections

    Returns
    -------
    f_int : ndarray shape (num_dof)
    K_T : ndarray shape (num_dof, num_dof)
    f_D : ndarray shape (num_dof)
        f_D = (B)^T D^\mu used to compute the homogenized tangent
    ep_flag : str
    int_var_iter : dict
        dictionary of interal state variables for each gauss point for
        each element

    Note
    ----
    Reference (Eq. 4.65 (1) Neto 2008)

    """
    num_dof = model.num_dof

    # initialize global vector and matrices
    f_int = np.zeros(num_dof)
    K_T = np.zeros((num_dof, num_dof))
    f_D = np.zeros((num_dof, 3))


    # initialize taylor constitutive tanget and homogenized stress
    D_taylor = np.zeros((3, 3))
    sig_hom = np.zeros(4)

    # new every Newton iteration
    # returned internal variables for specific
    # Delta_u_mic = C.T @ Delta_eps + Delta_u_fluct
    # and for a specific previous step "n"
    int_var_trial_mic = {'eps_e': {}, 'eps': {}, 'eps_bar_p': {},
                        'dgamma': {}, 'sig': {}, 'eps_p': {}, 'q': {}}

    # Loop over elements
    for eid, [etype, *edata] in model.elements.items():
        # create element object
        element = constructor(eid, etype, model)
        # recover element nodal displacement increment,  shape (8,)
        dof = np.array(element.dof) - 1  # numpy starts at 0
        Delta_u_fluct_ele = Delta_u_fluct[dof]

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
        f_D_e = np.zeros((8, 3))
        k_T_e = np.zeros((8, 8))

        # loop over quadrature points
        for gpid, [w, gp] in enumerate(zip(element.gauss.weights,
                                           element.gauss.points)):
            # build element strain-displacement matrix shape (3, 8)
            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)
            B = element.gradient_operator(dN_xi)

            # compute strain increment from
            # current displacement increment, shape (3, )
            Delta_eps = B @ Delta_u_fluct_ele
            # Add Delta_eps_macro for this step n -> n+1 Delta_eps_zz = 0
            Delta_eps = np.append(Delta_eps, 0) + Delta_eps_macro  

            # elastic trial strain
            # use the previous value stored for this element and this gp
            # TODO: how to apply the macroscale strain here DONE
            # Asada 2007 -> initial Delta_eps is arbitrary
            eps_e_trial = (int_var['eps_e_mic'][(Meid, Mgpid)][(eid, gpid)] +
                           Delta_eps)

            # trial accumulated plastic strain
            # this is only updated when converged
            eps_bar_p_trial = int_var['eps_bar_p_mic'][(Meid, Mgpid)][(eid,
                                                                       gpid)]
            # plastic strain trial is from previous load step
            eps_p_trial = int_var['eps_p_mic'][(Meid, Mgpid)][(eid, gpid)]

            # Step (7) algorithm 5
            # update internal variables for this gauss point
            sig, int_var_trial_mic, ep_flag = suvm(
                E, nu, H, sig_y0, eps_e_trial, eps_bar_p_trial,
                eps_p_trial, max_num_local_iter, int_var_trial_mic,
                eid, gpid)
            # Step (7) algorithm 5
            # use dgama from previous global iteration
            # int_var stores both micro and macro internal variabled
            D = consistent_tangent_mises(
                int_var['dgamma_mic'][(Meid, Mgpid)][(eid, gpid)],
                sig, E, nu, H, ep_flag)

            # Step (2) algorithm 7 thesis
            # volume average of all elements = sum over all gp and over all ele
            D_taylor += D * (dJ * w) / model.volume

            # Step (1) algorithm 6 thesis
            # sum over all gp and over all ele
            sig_hom += sig[:] * (dJ * w) / model.volume
            
            # Step (7) algorithm 5
            # compute element internal force (gaussian quadrature)
            # XFEM will affect here in the matrix B? I don't think so
            # sig[:3] ignore the 33 component here
            f_int_e += B.T @ sig[:3] * (dJ * w) # * element.thickness)

            # Step (8) algorithm 5
            # element consistent tanget matrix (gaussian quadrature)
            # XFEM will affect here? Yes
            # thickness will cancel out later, useful if it vary from ele
            k_T_e += B.T @ D @ B * (dJ * w) # * element.thickness)

            # Step (9) algorithm 5
            # this vector is used for compute the homogenized tangent
            f_D_e += B.T @ D * (dJ * w)

        # Build global matrices outside the quadrature loop
        # += because elements can share same dof
        f_int[element.id_v] += f_int_e
        K_T[element.id_m] += k_T_e
        f_D[element.id_v] += f_D_e

    return f_int, K_T, f_D, int_var_trial_mic, D_taylor, sig_hom
