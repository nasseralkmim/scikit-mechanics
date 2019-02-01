"""define the ramp function for weak descontinuity"""
import numpy as np


def ramp(phi, phi_i):
    """Shifted ramp function

     .. math:: /bar{/psi_i}(X) = |/phi(X)| + |phi(X_i)|

    where :math:`/psi(X)_i` is the shifted enriched function for node i
    and :math:`/phi(X)` is the level set function applied to spatial
    coordinates :math:`X=(x, y, z)`.

    Args:
        phi (array): level set function phi(x)
        phi_i (array): level set at node i

    Returns:
        float : ramp function
    """
    psi = np.abs(phi) - np.abs(phi_i)

    return psi


if __name__ == '__main__':
    pass
