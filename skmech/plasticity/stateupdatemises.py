"""Return-mapping algorithm considering von Mises"""
import numpy as np


def state_update_mises(E, nu, H, sig_y0,  # material properties
                       eps_e_trial, eps_bar_p_trial,  # internal variables
                       max_local_iteration=10):  # others
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
    From sec. 7.3.5 Neto 2008

    RSTAVA = {eps_e, eps_bar_e} so far are the STAte VAriables (other than
    tress components), but it could be expanded. This function should return
    those variables updated

    """
    # initialize internal variable for plastic multiplier variation
    dgama = 0.

    # material properties
    G = E / (2 * (1 + nu))      # shear modulus
    K = E / (3 * (1 - 2 * nu))  # bulk modulus

    # Elastic predictor step
    eps_v = eps_e_trial[0] + eps_e_trial[1]  # volumetric strain
    p = K * eps_v                       # hydrostatic stress
    # elastic trial deviatoric strain
    eps_d = eps_e_trial - (1 / 3) * eps_v * np.array([1, 1, 0])

    # convert engineering shear strain component into physical
    eps_d[2] = eps_d[2] / 2

    # von Mises effective stress
    J2 = (0.5) * (2 * G)**2 * (eps_d[0]**2 + eps_d[1]**2 + 2 * eps_d[2]**2)
    q_trial = np.sqrt(3 * J2)

    # set initial cummulative plastic strain using previously converged value
    # use set of internal variable at previous step (n)
    # no step indicates for update variable
    eps_bar_p = eps_bar_p_trial

    # hardening function
    # TODO; make ir more general later, (linear so far)
    sig_y = sig_y0 + H * eps_bar_p

    # yield function
    Phi = q_trial - sig_y

    if Phi >= 0:
        # plastic step
        # TODO: compare this NR with analytic
        # TODO: later add internal variables
        # sig, eps_e, eps_bar_e, dgama = local_constitutive_newton_raphson(
        #     max_local_iteration,
        #     dgama, Phi, eps_bar_p, q_trial,
        #     p, eps_d, eps_v,
        #     G, H, sig_y0)  # material parameters

        # first considering linear hardening function
        # TODO: use this to test the NR implementation above
        sig, eps_e, eps_bar_e, dgama = local_constitutive_linear_hardening(
            Phi, G, H, eps_d, q_trial, p, eps_v, eps_bar_p)

        # elastoplastic flag, if True plastic step
        elastoplastic_flag = True
    else:
        # elastic step
        sig = 2 * G * eps_d + p * np.array([1, 1, 0])
        eps_e = eps_e_trial
        # Note: eps_bar_p and dgama don't change in the elastic step

        # elastoplastic flag, if False elastic step
        elastoplastic_flag = False

    return sig, eps_e, eps_bar_p, dgama, elastoplastic_flag


def local_constitutive_linear_hardening(Phi, H, G, eps_d, q_trial, p, eps_v,
                                        eps_bar_p):
    """Solves the local constitutive equations with linear hardening function

    This is an analytic solution for the particular case considering linear
    hardening curve and von Mises yield surface. This is going to be compared
    with the Newton-Raphson procedure

    Returns
    -------
    eps_d : ndarray shape (3,)
        Deviatoric trial strain
    eps_bar_p : float
        Cummulative plastic strain
    dgama : float
        Change in the plastic ultiplier gamma

    Note
    ----
    Reference sec. 7.3.4 Neto 2008

    """
    # solution to the constitutive equation considering linear hardening
    dgama = Phi / (3 * G - H)

    # deviatoric stress trial
    sig_d_trial = 2 * G * eps_d

    # update deviatoric stress
    sig_d = (1 - dgama * 3 * G / q_trial) * sig_d_trial

    # update stress tensor
    sig = sig_d + p * np.array([1, 1, 0])

    # update elastic strain
    eps_e = 1 / (2 * G) * sig_d + (1 / 3) * eps_v * np.array([1, 1, 0])

    # update cummulative plastic strain
    eps_bar_p = eps_bar_p + dgama

    return sig, eps_e, eps_bar_p, dgama


def local_constitutive_newton_raphson(
        max_local_iteration,
        dgama, Phi, eps_bar_p, q_trial,
        p, eps_d, eps_v,
        G, H, sig_y0,  # material parameters
        tol=1e-6):
    """Solve the local constitutive equation for von Mises model

    Solve the single nonlinear equation considering the von Mises yield
    criterion.

    Reference Box 7.4 Neto 2008

    Parameters
    ----------
    max_local_iteration : int (default=10)
        Maximum number of iterations allowed for the local Newton-Raphson
        procedure.
    tol : float (default=1e-6)
        Tolerance for checking convergence of the Nwton-Raphson method.

    Returns
    -------
    sig : ndarray shape (3,)
    eps_e : ndarray shape (3,)
    eps_bar_p : float
    dgama : float

    """
    for k in range(1, max_local_iteration):
        # residual derivative with respect to dgama
        # the Phi function is already in a residual form
        # TODO: right now i'm considering only linear hardening H
        dPhi_ddgama = -3 * G - H

        # newton-raphson iteration
        dgama = dgama - Phi / dPhi_ddgama

        # new Phi (residual)
        eps_bar_p = eps_bar_p + dgama
        sig_y = sig_y0 + H * eps_bar_p
        # Phi Eq. 7.91 Neto 2008
        Phi = q_trial - 3 * dgama - sig_y

        # check convergence
        if abs(Phi / sig_y) <= tol:
            # update accumulated plastic strain -> alread updated eps_bar_p
            # update stress
            fact = (1 - 3 * dgama * G / q_trial)
            sig = 2 * G * fact * eps_d + p * np.array([1, 1, 0])

            # update elastic (engineering) strain
            eps_e = fact * eps_d + eps_v / 3 * np.array([1, 1, 0])
            eps_e[2] = eps_e[2] * 2  # convert back to engineering strain

            # break our of the NR loop
            break

    return sig, eps_e, eps_bar_p, dgama


if __name__ == '__main__':
    pass
