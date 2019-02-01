"""Computes a single increment"""
import numpy as np
from .localization import localization
from ..postprocess.saveoutput import save_output
from .partitioned import solve_partitioned


def increment_step(model, u, int_var, f_ext, max_num_iter,
                   max_num_local_iter, increment, tol, lmbda, start,
                   element_out, node_out):
    """Perform an increment step on the analysis

    Parameters
    ----------
    u : ndarray
    int_var : dict
    f_ext : ndarray
    max_num_iter : int
        number of iterations for the newton raphson method, rule of thumb is 8
        (Borst 2012)

    Returns
    -------
    u : ndarray
        updated displacement with converged value u_{n+1} = u_{n} + Delta_u
    int_var : dict
        updated internal variables, only when converged

    """
    # initial displacement increment for each load step
    # will accumulate the corrections for each step n -> n+1
    Delta_u = np.zeros(model.num_dof)

    # Step (2), (3) algorithm 1 master thesis
    # int_var_trial internal variable dict that will be updated during iteration
    # int_var_trial will be used after convergence of NR to save internal variables
    f_int, K_T, int_var_trial = localization(model, Delta_u,
                                            int_var,
                                            max_num_local_iter,
                                            increment, start, lmbda)

    # residual for first iteration unbalanced force
    r_first = f_int - f_ext
    r_norm_first = np.linalg.norm(r_first)

    # Begin global Newton procedures
    for k in range(0, max_num_iter + 1):
        # Step (4) Assemble global and solve for correction
        newton_correction, f_ext = solve_partitioned(
            model, K_T, f_int, f_ext, increment, k)

        # Step (5) Update solutions algorithm 1 master thesis
        Delta_u += newton_correction
        u += newton_correction

        # Step (6) (7) (8) algorithm 1 master thesis
        # build internal load vector and solve local constitutive equation
        f_int, K_T, int_var_trial = localization(model, Delta_u,
                                                int_var,
                                                max_num_local_iter,
                                                increment, start, lmbda)
        # new residual
        r_updt = f_int - f_ext
        r_norm = np.linalg.norm(r_updt)

        # check convergece criterion from Borst 2012 eq. 4.66
        criterion = r_norm <= tol * r_norm_first

        print(f'Iteration {k + 1} residual norm {r_norm:.1e}')
        if criterion:
            # solution converged +1 because it started in 0
            print(f'Converged with {k + 1} iterations '
                  f'residual norm {r_norm:.1e}')

            # after solution converged update internal variable dict
            # this will be used in the next time step
            int_var = update_int_var(int_var, int_var_trial, model)
            save_output(model, u, int_var, increment, start, lmbda,
                        element_out, node_out)

            break
        else:
            # did't converge, continue to next global iteration
            continue
    else:
        raise Exception(f'Solution did not converge at time step '
                        f'{increment + 1} after {k} iterations')

    return u, int_var


def update_int_var(int_var, int_var_trial, model):
    """Update internal variables with iteration values"""
    # TODO: update interal variables converged DONE
    int_var['eps_e'] = int_var_trial['eps_e'].copy()
    int_var['eps_p'] = int_var_trial['eps_p'].copy()
    int_var['eps'] = int_var_trial['eps'].copy()
    int_var['eps_bar_p'] = int_var_trial['eps_bar_p'].copy()
    int_var['dgamma'] = int_var_trial['dgamma'].copy()
    int_var['sig'] = int_var_trial['sig'].copy()
    int_var['q'] = int_var_trial['q'].copy()

    if model.micromodel is not None:
        int_var['eps_e_mic'] = int_var_trial['eps_e_mic'].copy()
        int_var['eps_p_mic'] = int_var_trial['eps_p_mic'].copy()
        int_var['eps_mic'] = int_var_trial['eps_mic'].copy()
        int_var['eps_bar_p_mic'] = int_var_trial['eps_bar_p_mic'].copy()
        int_var['dgamma_mic'] = int_var_trial['dgamma_mic'].copy()
        int_var['sig_mic'] = int_var_trial['sig_mic'].copy()
        int_var['q_mic'] = int_var_trial['q_mic'].copy()
    return int_var
