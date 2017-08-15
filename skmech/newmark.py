from . import boundary
from  import stiffness
from . import load
from . import traction
from . import stress
from . import mass
from . import strain
import numpy as np


def coefficients(dt, beta, gamma):
    """Generates the coefficients used in the procedure

    """
    c0 = 1/(beta * dt**2)
    c2 = 1/(beta * dt)
    c3 = 1/(2 * beta) - 1
    c6 = dt*(1 - gamma)
    c7 = gamma*dt
    return c0, c2, c3, c6, c7


def iterations(model, material, N, dt, gamma, beta, U0, V0, b_force,
               trac_bc, displ_bc, temp_bc, flux_bc, b_heat, T0):
    """Perform N iterations using Newmark's method

    """
    # initiate the arrays with results for each time step
    U = np.zeros((model.ndof, N))
    SIG = np.zeros((model.nn, 3, N))

    # Generate initial values
    U[:, 0], V_t = initial_values(U0, V0, model)
    K, M, P, EPS0, T0 = generate_fem(model, material, b_force, trac_bc,
                                     temp_bc, flux_bc, b_heat, T0, t=0)

    A_t = np.linalg.solve(M, (P - K @ U[:, 0]))
    SIG[:, :, 0] = stress.recovery(model, material, U[:, 0], EPS0)

    for n in range(1, N):
        t = n*dt
        K, M, P, EPS0, T0 = generate_fem(model, material, b_force, trac_bc,
                                         temp_bc, flux_bc, b_heat, T0, t)
        c0, c2, c3, c6, c7 = coefficients(dt, beta, gamma)

        # Compute matrices of the system K_ U_updt = P_
        K_ = K + c0 * M
        P_ = P + M @ (c0 * U[:, n-1] + c2 * V_t + c3 * A_t)
        K_m, P_m = boundary.dirichlet(K_, P_, model, displ_bc)

        # Calculate at time t + dt
        U[:, n] = np.linalg.solve(K_m, P_m)
        A_updt = c0*(U[:, n] - U[:, n-1]) - c2*V_t - c3*A_t
        V_updt = V_t + c6*A_t + c7*A_updt
        SIG[:, :, n] = stress.recovery(model, material, U[:, n], EPS0)

        # Update the variables that are not stored
        V_t = V_updt
        A_t = A_updt

    return U, SIG


def initial_values(U0, V0, model):
    """Return the initial values if given

    """
    if np.size(U0) == 1:
        U = np.zeros(model.ndof)
        V = np.zeros(model.ndof)
    else:
        U = U0
        V = V0
    return U, V


def generate_fem(model, material, b_force, trac_bc,
                 temp_bc, flux_bc, b_heat, T0, t):
    """Generate the matrices and vectors for FEM calculations

    """
    K = stiffness.K_matrix(model, material, t)

    M = mass.M_matrix(model, material, t)

    Pb = load.Pb_vector(model, b_force, t)

    Pt = traction.Pt_vector(model, trac_bc, t)

    if b_heat and flux_bc and temp_bc is not None:
        # Tu is the updated temperature on each node
        EPS0, Tu = strain.temperature(model, material, temp_bc, flux_bc,
                                      b_heat, T0, t)
    else:
        EPS0 = np.zeros((model.ne, 3))
        Tu = np.zeros(model.nn)

    Pe = load.Pe_vector(model, material, EPS0, t)

    P = Pb + Pt + Pe

    return K, M, P, EPS0, Tu
