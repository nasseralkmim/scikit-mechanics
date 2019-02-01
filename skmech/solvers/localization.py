"""Perform the localization procedure of FEM

Localization means to loop over elements, build element matrices and assemble
global

"""
import numpy as np
from ..constructor import constructor
from ..plasticity.stateupdatemises import state_update_mises as suvm
from ..plasticity.tangentmises import consistent_tangent_mises
from ..multiscale.microincremental import micro_incremental
from ..postprocess.saveoutput import save_output


def localization(model, Delta_u, int_var, max_num_local_iter,
                 increment, start, lmbda):
    """Localization of fem procedure

    Parameters
    ----------
    model : Model object
    num_dof : number of degree's of freedom
    Delta_u : displacement increment
    eps_e_n : dict {(eid, gpid): ndarray shape (4)}
        Stores the elastic strain at previous step for each element and each
        gauss point. This value is updated every time this function is called.
        Note that it has 4 components, the elastic strain component, eps_e_33,
        is not zero
    eps_p_n : dict {(eid, gpid): ndarray shape (4)}
        Note that it has 4 components, the plastic strain component, eps_e_33,
        is not zero, even though the total strain eps_33 = 0 for plane strain.
    eps_bar_p_n : dict {(eid, gpid): float}
        Stores the accumulated plastic strain at previous step for each element
        and each gauss point (gp). This value is updated every time this
        function is called
    dgamma_n : dict {(eid, gpid): float}

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

    Find the stress for a given displacement u, then multiply the stress for
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

    Note
    ----
    If micromodel option is True in the incremental procedure then substitute
    the state update (suvm) module with an microscale solver.

    """
    num_dof = model.num_dof
    # initialize global vector and matrices
    f_int = np.zeros(num_dof)
    K_T = np.zeros((num_dof, num_dof))

    # dictionary with local variables for this iteration
    # new every local Newton iteration
    # use to save converged value
    int_var_trial = {'eps_e': {}, 'eps': {}, 'eps_bar_p': {},
                     'dgamma': {}, 'sig': {}, 'eps_p': {}, 'q': {}}

    if model.micromodel is not None:
        # every localization step we need to reset internal micro
        # variabls
        # save only after macro solution converged
        int_var_trial = model.micromodel.set_internal_var(
            int_var_trial , model)

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
        for gpid, [w, gp] in enumerate(zip(element.gauss.weights,
                                           element.gauss.points)):
            # build element strain-displacement matrix shape (3, 8)
            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)
            # TODO: XFEM affects here, reqwrite the method for the enriched
            # element class
            B = element.gradient_operator(dN_xi)

            # compute strain increment from
            # current displacement increment, shape (3, )
            Delta_eps = B @ Delta_u_ele
            Delta_eps = np.append(Delta_eps, 0)  # Delta_eps_zz = 0

            # elastic trial strain
            # use the previous value stored for this element and this gp
            eps_e_trial = int_var['eps_e'][(eid, gpid)] + Delta_eps

            # trial accumulated plastic strain
            # this is only updated when converged
            eps_bar_p_trial = int_var['eps_bar_p'][(eid, gpid)]

            # plastic strain trial is from previous load step
            eps_p_trial = int_var['eps_p'][(eid, gpid)]
            
            if model.micromodel is not None:
                # Multiscale analysis for this element for this gauss point
                print(f'Element {eid} Gauss Point {gpid} ', end='')
                sig, D, int_var_trial = micro_incremental(
                    model.micromodel, Delta_eps,
                    int_var_trial,
                    int_var,
                    max_num_local_iter,
                    eid, gpid)

                # scale transition
                # update macro internal variables using micro variables
                # int_var_trial will update int_var only when macro converge
                int_var_trial = update_macro_variables(int_var_trial,
                                                       sig, D, nu, eid, gpid)

                # save the macrostrain for this Newton itaration
                # this will be used when the macro solution converged so we
                # can compute the micro displacement field
                # only keep the last one which is the converged value
                model.micromodel.Delta_eps_mac[(eid, gpid)] = Delta_eps
            else:
                # update internal variables for this gauss point
                sig, int_var_trial, ep_flag = suvm(
                    E, nu, H, sig_y0, eps_e_trial, eps_bar_p_trial,
                    eps_p_trial, max_num_local_iter, int_var_trial,
                    eid, gpid)

                # use dgama from previous global load step!
                D = consistent_tangent_mises(
                    int_var['dgamma'][(eid, gpid)], sig, E, nu, H, ep_flag)

            # compute element internal force (gaussian quadrature)
            # sig[:3] ignore the 33 component here
            # ignore thickness, considering constant over all elements
            # in the xfem there will be a Benr and Bstd
            f_int_e += B.T @ sig[:3] * (dJ * w) # * element.thickness)

            # element consistent tanget matrix (gaussian quadrature)
            k_T_e += B.T @ D @ B * (dJ * w) # * element.thickness)

        # Build global matrices outside the quadrature loop
        # += because elements can share same dof
        f_int[element.id_v] += f_int_e
        K_T[element.id_m] += k_T_e

    return f_int, K_T, int_var_trial



def update_macro_variables(int_var_trial, sig, D, nu, eid, gpid):
    """Update macro variables with micro results """
    int_var_trial['sig'][(eid, gpid)] = sig
    # add zz component
    # if len(int_var_trial['sig'][(eid, gpid)]) != 4:
    #     int_var_trial['sig'][(eid, gpid)] = np.append(
    #         int_var_trial['sig'][(eid, gpid)], 0)
    # # zz component for plane strain = nu (sig11 + sig22)
    # int_var_trial['sig'][(eid, gpid)][3] = 0.25 * (sig[0] + sig[1])

    # TODO this seems to be wrong, How to compute macro strain DONE It is right
    # use sig = D : eps with homogenized tangent and homogenized stress
    # D is the tangent constitutive operator
    int_var_trial['eps'][(eid, gpid)] = np.linalg.solve(D, sig[:3])
    if len(int_var_trial['eps'][(eid, gpid)]) != 4:
        int_var_trial['eps'][(eid, gpid)] = np.append(
            int_var_trial['eps'][(eid, gpid)], 0)

    int_var_trial['eps_e'][(eid, gpid)] = int_var_trial['eps'][(eid, gpid)]
    if len(int_var_trial['eps_e'][(eid, gpid)]) != 4:
        int_var_trial['eps_e'][(eid, gpid)] = np.append(
            int_var_trial['eps_e'][(eid, gpid)], 0)

    # Assume that the macroscale is elastic
    int_var_trial['eps_bar_p'][(eid, gpid)] = 0
    int_var_trial['eps_p'][(eid, gpid)] = np.zeros(3)

    return int_var_trial
