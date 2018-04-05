"""Computes a single increment"""
import numpy as np
from .localization import localization
from ..postprocess.saveoutput import save_output
from .partitioned import solve_partitioned


def increment_step(model, u, int_var, f_ext, max_num_iter, max_num_local_iter,
                   increment, tol, lmbda, start, element_out, node_out):
    """Perform an increment step on the analysis

    Parameters
    ----------
    u : ndarray
    int_var : dict
    f_ext : ndarray

    Returns
    -------
    u : ndarray
        updated displacement with converged value u_{n+1} = u_{n} + Delta_u
    int_var : dict
        updated internal variables, only when converged

    """
    # initial displacement increment for each load step
    Delta_u = np.zeros(model.num_dof)
    # Step (2), (3)
    f_int, K_T, int_var_iter = localization(model, Delta_u,
                                            int_var,
                                            max_num_local_iter)
    # Begin global Newton procedures
    for k in range(0, max_num_iter + 1):
        # Step (4) Assemble global and solve for correction
        newton_correction, f_ext = solve_partitioned(
            model, K_T, f_int, f_ext, increment, k)

        # Step (5) Update solutions
        Delta_u += newton_correction
        u += newton_correction

        # Step (6) (7) (8)
        # build internal load vector and solve local constitutive equation
        f_int, K_T, int_var_iter = localization(model, Delta_u,
                                                int_var,
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

            int_var = update_int_var(int_var, int_var_iter, model)
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


def update_int_var(int_var, int_var_iter, model):
    """Update internal variables with iteration values"""
    # TODO: update interal variables converged DONE
    int_var['eps_e'] = int_var_iter['eps_e'].copy()
    int_var['eps_p'] = int_var_iter['eps_p'].copy()
    int_var['eps'] = int_var_iter['eps'].copy()
    int_var['eps_bar_p'] = int_var_iter['eps_bar_p'].copy()
    int_var['dgamma'] = int_var_iter['dgamma'].copy()
    int_var['sig'] = int_var_iter['sig'].copy()
    int_var['q'] = int_var_iter['q'].copy()

    if model.micromodel is not None:
        int_var['eps_e_mic'] = int_var_iter['eps_e_mic'].copy()
        int_var['eps_p_mic'] = int_var_iter['eps_p_mic'].copy()
        int_var['eps_mic'] = int_var_iter['eps_mic'].copy()
        int_var['eps_bar_p_mic'] = int_var_iter['eps_bar_p_mic'].copy()
        int_var['dgamma_mic'] = int_var_iter['dgamma_mic'].copy()
        int_var['sig_mic'] = int_var_iter['sig_mic'].copy()
        int_var['q_mic'] = int_var_iter['q_mic'].copy()
    return int_var
