"""Return-mapping algorithm considering von Mises"""
import numpy as np


def state_update_mises(E, nu, H, sig_y0,
                       eps_e_trial, eps_bar_p_trial, eps_p_trial,
                       max_num_local_iter, material_case,
                       int_var, eid, gpid):
    """State update procedure considering von Mises yield criterion

    Update variables by solving the nonlinear constitutive equations using
    a predictor/return-mapping algorithm

    eps_e_trial is the elastic trial strain computed using the elastric strain
    converged in the previous step (n) and the strain increment computed with
    the global Newton-Raphson displacement increment

    Parameters
    ----------
    eps_e_trial : ndarray shape (3,)
        Elastric trial strain obtained from adding the strain increment (global
        Newton-Raphson solution) to the previously elastic strain obtained from
        the state update module
    eps_bar_p_trial : float
        Cummulative plastic strain obtained from the previous load increment

    Returns
    -------

    Note
    ----
    From (sec. 7.3.5 Neto 2008)

    RSTAVA = {eps_e, eps_bar_e} so far are the STAte VAriables (other than
    tress components), but it could be expanded. This function should return
    those variables updated

    """
    # initializa plastic multiplier variation
    dgamma = 0

    # material properties
    G = E / (2 * (1 + nu))      # shear modulus
    K = E / (3 * (1 - 2 * nu))  # bulk modulus

    # Elastic predictor step
    # volumetric strain (eq. 3.90 Neto 2008)
    # Note the eps_e_33 component, beucase uven though the eps_33 = 0 the
    # elastic part is not zero
    eps_v_trial = eps_e_trial[0] + eps_e_trial[1] + eps_e_trial[3]
    p = K * eps_v_trial                       # hydrostatic stress
    # elastic trial deviatoric strain
    eps_d_trial = eps_e_trial - (1 / 3) * eps_v_trial * np.array([1, 1, 0, 1])
    # convert engineering shear strain component into physical
    eps_d_trial[2] = eps_d_trial[2] / 2

    eps_p = eps_p_trial

    # von Mises effective stress
    sig_d_trial = 2 * G * eps_d_trial
    J2 = (0.5) * (sig_d_trial[0]**2 + sig_d_trial[1]**2 +
                  2 * sig_d_trial[2]**2 + sig_d_trial[3]**2)
    q_trial = np.sqrt(3 * J2)

    # print(f'qtrial {q_trial:.1e} eps_e_trail {eps_e_trial}')
    # set initial cummulative plastic strain using previously converged value
    # use set of internal variable at previous step (n)
    # no step indicates for update variable
    eps_bar_p = eps_bar_p_trial

    # hardening function
    # TODO; make ir more general later, (linear so far)
    sig_y = sig_y0 + H * eps_bar_p

    # yield function
    Phi_trial = q_trial - sig_y

    if Phi_trial > 0:
        # plastic step
        # TODO: compare this NR with analytic
        # TODO: later add internal variables
        local_newton = False
        if local_newton is True:
            sig, eps_e, eps_p, eps_bar_p, dgamma, q = local_constit_newton(
                max_num_local_iter,
                Phi_trial, eps_bar_p, q_trial,
                p, eps_d_trial, eps_v_trial, eps_p,
                G, H, sig_y0)  # material parameters
        else:
            # first considering linear hardening function
            # TODO: use this to test the NR implementation above
            sig, eps_e, eps_p, eps_bar_p, dgamma, q = local_const_lin_hard(
                Phi_trial, G, H, eps_d_trial, q_trial, p, eps_v_trial,
                eps_bar_p, eps_p)
        # elastoplastic flag, if True plastic step
        ep_flag = True

    else:
        # elastic step
        eps_e = eps_e_trial
        eps_p = eps_p_trial     # don't change plastic strain
        sig = 2 * G * eps_d_trial + p * np.array([1, 1, 0, 1])
        q = q_trial
        # Note: eps_bar_p and dgamma don't change in the elastic step
        # elastoplastic flag, if False elastic step
        ep_flag = False

    int_var_updt = storage_int_var(int_var, eid, gpid, eps_e, eps_p, sig,
                                   eps_bar_p, q, dgamma)

    return sig, int_var_updt, ep_flag


def storage_int_var(int_var, eid, gpid, eps_e, eps_p, sig,
                    eps_bar_p, q, dgamma):
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
    # the tuple (eid, gpid) for each element and each gauss point

    # first copy the int_var dict, so the internal variables is changed
    # only when the solution converged
    int_var_updt = int_var.copy()
    int_var_updt['eps_e'][(eid, gpid)] = eps_e
    int_var_updt['eps_p'][(eid, gpid)] = eps_p
    int_var_updt['eps'][(eid, gpid)] = eps_e + eps_p
    int_var_updt['eps_bar_p'][(eid, gpid)] = eps_bar_p
    int_var_updt['dgamma'][(eid, gpid)] = dgamma
    int_var_updt['sig'][(eid, gpid)] = sig
    int_var_updt['q'][(eid, gpid)] = q

    return int_var_updt


def local_const_lin_hard(Phi_trial, G, H, eps_d_trial, q_trial,
                         p, eps_v_trial,
                         eps_bar_p, eps_p):
    """Solves the local constitutive equations with linear hardening function

    This is an analytic solution for the particular case considering linear
    hardening curve and von Mises yield surface. This is going to be compared
    with the Newton-Raphson procedure

    Returns
    -------
    eps_d_trial : ndarray shape (3,)
        Deviatoric trial strain
    eps_bar_p : float
        Cummulative plastic strain
    dgamma : float
        Change in the plastic multiplier gamma

    Note
    ----
    Reference (sec. 7.3.4 Neto 2008)

    """
    # solution to the constitutive equation considering linear hardening
    dgamma = Phi_trial / (3 * G + H)   # Eq. 7.101

    # deviatoric stress trial
    sig_d_trial = 2 * G * eps_d_trial

    # update deviatoric stress, not trial anymore
    sig_d = (1 - dgamma * 3 * G / q_trial) * sig_d_trial

    sig_d_norm = np.sqrt(sig_d[0]**2 + sig_d[1]**2 + sig_d[3]**2 +
                         2 * sig_d[2]**2)
    q = np.sqrt(3 / 2) * sig_d_norm

    # update stress tensor
    sig = sig_d + p * np.array([1, 1, 0, 1])

    # update elastic strain
    eps_e = (1 / (2 * G) * sig_d +
             (1 / 3) * eps_v_trial * np.array([1, 1, 0, 1]))
    # convert back to engineering strain
    eps_e[2] = 2 * eps_e[2]

    # update cummulative plastic strain
    eps_bar_p = eps_bar_p + dgamma

    eps_p = eps_p + dgamma * np.sqrt(3 / 2) * sig_d / sig_d_norm

    return sig, eps_e, eps_p, eps_bar_p, dgamma, q


def local_constit_newton(max_num_local_iter,
                         Phi_trial, eps_bar_p, q_trial,
                         p, eps_d_trial, eps_v_trial, eps_p,
                         G, H, sig_y0,  # material parameters
                         tol=1e-6):
    """Solve the local constitutive equation for von Mises model

    Solve the single nonlinear equation considering the von Mises yield
    criterion.

    Reference (Box 7.4 Neto 2008)

    Parameters
    ----------
    max_num_local_iter : int (default=10)
        Maximum number of iterations allowed for the local Newton-Raphson
        procedure.
    tol : float (default=1e-6)
        Tolerance for checking convergence of the Nwton-Raphson method.

    Returns
    -------
    sig : ndarray shape (3,)
    eps_e : ndarray shape (3,)
    eps_bar_p : float
    dgamma : float
    q : float
        von Mises effective stress

    """
    dgamma = 0
    for k in range(1, max_num_local_iter):
        # residual derivative with respect to dgamma
        # the Phi function is already in a residual form
        # TODO: right now i'm considering only linear hardening H
        dPhi_ddgama = -3 * G - H

        # newton-raphson iteration
        dgamma = dgamma - Phi_trial / dPhi_ddgama

        # new Phi (residual)
        # eps_bar_p_(n+1) = eps_bar_p_(n) + Delta_gamma,
        # (- Phi / dPhi_ddgama) represents the change
        # in Delta_gamma. See (Neto 2008 page 226 line 71)
        eps_bar_p = eps_bar_p + (- Phi_trial / dPhi_ddgama)
        sig_y = sig_y0 + H * eps_bar_p

        # Phi (Eq. 7.91 Neto 2008)
        Phi = q_trial - 3 * G * dgamma - sig_y

        # check convergence
        if abs(Phi / sig_y) <= tol:
            # update accumulated plastic strain -> alread updated eps_bar_p
            # update stress
            fact = (1 - 3 * dgamma * G / q_trial)
            # 2 D eps_d_trial is trial deviatoric strss
            sig_d = fact * 2 * G * eps_d_trial
            sig_d_norm = np.sqrt(sig_d[0]**2 + sig_d[1]**2 + sig_d[3]**2 +
                                 2 * sig_d[2]**2)
            q = np.sqrt(3 / 2) * sig_d_norm
            sig = sig_d + p * np.array([1, 1, 0, 1])

            # update elastic (engineering) strain
            eps_e = (fact * eps_d_trial +
                     (1 / 3) * eps_v_trial * np.array([1, 1, 0, 1]))
            eps_e[2] = eps_e[2] * 2  # convert back to engineering strain

            # (eq. 7.87 Neto 2008)
            eps_p = eps_p + dgamma * np.sqrt(3 / 2) * sig_d / sig_d_norm
            return sig, eps_e, eps_p, eps_bar_p, dgamma, q
            # break our of the NR loop
            break
    else:
        raise Exception('Local Newton-Raphson did not converge')


if __name__ == '__main__':
    pass
