"""Build the consistent tangent matrix"""
import numpy as np


def consistent_tangent_mises(dgama, sig, E, nu, H, elastoplastic_flag):
    """Build the consistent tangent matrix consideirng von Mises (ctvm)

    Parameters
    ----------
    dgama : float
        Incremental plastic multiplier obtained from the previous global
        equilibrium iteration (global Newton-Raphson)
    elastoplastic_flag : boolean
        If true return elastoplastic tangent matrix, if false returns the
        elastic tangent matrix
    sig : ndarray shape(3,)
        Updated stress components obtained from the nonlinear constitutive
        equations (suvm)
    H : float
        Hardening modulus (first considering linear hardening curve, do others
        latter). If the discrete hardening curve is considered, it will be
        required to pass the current eps_bar_p (cummulative plastic strain) in
        order to retrieve the approximate hardening slope H (varying)
    E, nu : float, float
        Elastic material properties

    Returns
    -------
    D : numpy array shape(8, 8)
        Consistent tangent matrix or elastic matrix depending on the
        elastoplastic flag

    Note
    ----
    Only plane-strain.

    Reference: section 7.4.2 Neto 2008.

    """
    # local Is fourth order symmetric identity tensor in equivalent matrix form
    fourth_sym_I = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, .5]])
    # local second order identity tensor in vector form
    second_i = np.array([1, 1, 0])
    # deviatoric projection tensor Eq. 3.94 p. 59 Neto 2008
    dev_sym_proj = fourth_sym_I - (1 / 3) * np.outer(second_i, second_i)

    # Material properties
    G = E / (2 * (1 + nu))      # shear modulus
    K = E / (3 * (1 - 2 * nu))  # bulk modulus

    # Compute consistent tangent
    p = (1 / 3) * (sig[0] + sig[1])         # hydrostatic stress
    s = sig - p * np.array([1, 1, 0])      # deviatoric

    # note that the factor 2 is due symmetry of deviatoric stress tensor
    s_norm = np.sqrt(s[0] * s[0] + s[1] * s[1] + 2 * s[2] * s[2])

    # equivalent von Mises
    # using the converged stress value to compute trial
    q_trial = np.sqrt(3 / 2) * s_norm + 3 * G * dgama

    if elastoplastic_flag is True:
        # Factors in Eq. 7.120 Neto 2008
        Afactor = 2 * G * (1 - 3 * G * dgama / q_trial)
        # TODO: considering only linear hardening curve with modulus H
        Bfactor = (6 * G**2 *
                   (dgama / q_trial - 1 / (3 * G + H)) / s_norm**2)
        # consistent tangent modulus Eq. 7.120 Neto 2008
        D = (Afactor * dev_sym_proj +
             Bfactor * np.outer(s, s) +
             K * np.outer(second_i, second_i))
    else:
        # elastic consistent tangent modulus Eq. 4.51 & 7.107 Neto 2008
        # equation 7.107 is only for plane strain
        if material_case == 'strain':
            D = 2 * G * dev_sym_proj + K * np.outer(second_i, second_i)
        if material_case == 'stress':
            D = 2 * G * fourth_sym_I + (K - 2 / 3 * G) * (
                (2 * G / (K + 4 / 3 * G)) * np.outer(second_i, second_i))
    return D


if __name__ == '__main__':
    dgama, E, nu, sig, H = .001, 1e9, .3, np.array([1, 1, 1]), 1e7
    D = consistent_tangent_mises(dgama, sig, E, nu, H,
                                 elastoplastic_flag=False)
    print(D)
