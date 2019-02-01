"""Solve the microscale problem using the global strain increment"""
import numpy as np
from .microlocalization import micro_localization
from ..postprocess.saveoutput import save_output


def micro_incremental(micromodel, Delta_eps, int_var_trial,
                      int_var, max_num_local_iter, Meid, Mgpid,
                      tol=5e-3):
    """Microscale solver

    This module is the main responsible for the microscale analysis

    Sequence:
        -> micro_incremental
            '-> micro_localization
                '-> element loop
                    '-> gauss point loop
                        '-> suvm, ctvm for each gauss point

    Parameters
    ----------
    Delta_eps : numpy array (4,)
        macro strain increment for a newton iteration
    int_var_trial : dict 
        dictionary with internal variables for macro scale Newton
        iteration, use this to update the internal variables only when
        solution convergenames as string keys, and other dictionaries
        as values. This contains the macroscale and microscale
        internal variables, this module updates the micro variables,
        the macro variabled are updated in the localization module
    int_var : dict
        contain initial values 
    Meid, Mgpid : int, int
        macro element id and macro gauss point id

    Returns
    -------
    sig : ndarray shape(3,)
        homogenized stress for a macroscale gauss point
    D : ndarray shape(3, 3)
        homogenized material stiffness for a gauss point
    int_var_mic : dict
        contains the internal variables for both macro and microscale

    Note
    ----
    TODO: currently the incremental procedure is not subdivided
    there is only one time increment [t_n, t_n+1], for nonlinear micro structure
    it may cause problems
    """
    # array that will accumulate the Newton corrections for fluctuations
    # num_dof_reg to not count enriched dofs
    Delta_u_fluct = np.zeros(micromodel.num_dof_reg)

    # Step (2) algorithm 5 thesis
    # localization computes new internal variables
    # those are stored only when solution converges
    # TODO: passa Meid and Mgpid to localization 
    f_int, K_T, f_D, int_var_trial_mic, _, _ = micro_localization(
        micromodel, Delta_u_fluct, int_var, Delta_eps, Meid, Mgpid,
        max_num_local_iter)

    r_norm_first = np.linalg.norm(f_int)

    # Begin Newton procedure
    for k in range(0, max_num_local_iter + 1):
        # Step (4)
        newton_correction, f_int_react = solve_partitioned_linear_bc(
            micromodel, K_T, f_int)

        # Step (5) algorithm 5 thesis
        Delta_u_fluct += newton_correction

        # Step (6, 7, 8) algorithm 5 thesis
        f_int, K_T, f_D, int_var_trial_mic, D_taylor, sig_hom = micro_localization(
            micromodel, Delta_u_fluct, int_var, Delta_eps, Meid, Mgpid,
            max_num_local_iter)

        # Check self equilibrium
        r_norm = np.linalg.norm(f_int)
        correction_norm = np.linalg.norm(newton_correction)
        criterion = correction_norm <  1e-12 

        if criterion:
            # solution converged +1 because it started in 0
            print(f'Microscale converged with {k + 1} iterations '
                  f'correction norm {correction_norm:.1e}')
            # update converged internal microscale variables
            int_var_trial = update_int_var_mic(int_var_trial,
                                               int_var_trial_mic,
                                               Meid, Mgpid)
            # sig, D = compute_homogenized(micromodel, K_T, f_int, int_var_trial)

            # save Delta_u_fluct for adding to micro displacement latter
            # when macro problem converge
            # Delta_u_fluct is associated with Delta_eps_mac
            # save for each Macro element and Macro gp id
            micromodel.Delta_u_fluct[(Meid, Mgpid)] = Delta_u_fluct
            break
        else:
            # not converged
            continue
    else:
        raise Exception(f"Micro solution did't converge after {k} iterations")
    return sig_hom, D_taylor, int_var_trial


def compute_homogenized(micromodel, K_T, f_int, int_var_trial):
    """Compute homogenized quantities"""
    V = micromodel.volume
    i, b = micromodel.id_i, micromodel.id_b
    ii = np.ix_(i, i)
    ib = np.ix_(i, b)
    bi = np.ix_(b, i)
    bb = np.ix_(b, b)
    Cb = micromodel.C_b
    Cg = micromodel.C_g
    KB = np.block([K_T[bi], K_T[bb]])
    KI = np.block([K_T[ii], K_T[ib]])
    Kii_inv = np.linalg.inv(K_T[ii])
    sig_test = 1 / V * Cb @ f_int[b]
    Klin = (KB - K_T[bi] @ Kii_inv @ KI)  # condensed linear stiffness
    D_test = 1 / V * Cb @ Klin @ Cg.T

    return sig, D


def solve_partitioned_linear_bc(micromodel, K_T, f_int):
    """Solve partitioned system of equations for microscale problem

    Obtain Newton correction for internal degree's of freedom and
    obtain residual for boundary degree's of freedom

    [ Kii Kib ] [ delta_u_i ] = - [ f_int_i ]
    [ Kbi Kbb ] [ delta_u_b ] = - [ f_int_b ]

    Returns
    -------
    delta_u_fluct, f_int_react
        newton correction (for displacement fluctuation) and internal force
        load for boundary dofs
    """
    i, b = micromodel.id_i, micromodel.id_b
    ii = np.ix_(i, i)
    bi = np.ix_(b, i)

    # zeros because linear displacement is assumed
    delta_u_fluct = np.zeros(micromodel.num_dof)

    delta_u_fluct[i] = np.linalg.solve(K_T[ii], - f_int[i])

    # reaction on boundary dofs
    f_int[b] = - K_T[bi] @ delta_u_fluct[i]

    return delta_u_fluct, f_int


def update_int_var_mic(int_var_trial, int_var_trial_mic, Meid, Mgpid):
    """Update internal variables with iteration values"""
    # the int_var_trial_mic comes from the suvm module, it is used in the
    # macroscale as well thats is why there is no suffix "_mic"
    int_var_trial['eps_e_mic'][(Meid, Mgpid)] = int_var_trial_mic['eps_e'].copy()
    int_var_trial['eps_p_mic'][(Meid, Mgpid)] = int_var_trial_mic['eps_p'].copy()
    int_var_trial['eps_mic'][(Meid, Mgpid)] = int_var_trial_mic['eps'].copy()
    int_var_trial['eps_bar_p_mic'][(Meid, Mgpid)] = int_var_trial_mic['eps_bar_p'].copy()
    int_var_trial['dgamma_mic'][(Meid, Mgpid)] = int_var_trial_mic['dgamma'].copy()
    int_var_trial['sig_mic'][(Meid, Mgpid)] = int_var_trial_mic['sig'].copy()
    int_var_trial['q_mic'][(Meid, Mgpid)] = int_var_trial_mic['q'].copy()
    return int_var_trial
