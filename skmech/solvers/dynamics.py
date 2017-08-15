from .. import newmark


def solver(model, material, b_force, trac_bc, displ_bc, t_int, dt,
           t=1, gamma=0.5, beta=0.25, U0=0, V0=0, plot_u=False,
           plot_s=False, vmin=None, vmax=None, magf=1, temp_bc=None,
           flux_bc=None, T0=0, b_heat=None):
    """Solves the elasto-dynamics problem

    """
    # Number of steps
    N = int(t_int/dt)+1

    U, SIG = newmark.iterations(model, material, N, dt, gamma, beta, U0,
                                V0, b_force, trac_bc, displ_bc,
                                temp_bc, flux_bc, b_heat, T0)

    return U, SIG
