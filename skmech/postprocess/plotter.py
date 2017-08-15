import matplotlib.pyplot as plt
from elastopy.postprocess import draw
from elastopy import stress
import matplotlib.animation as animation
import numpy as np


def show():
    plt.show()


def initiate(aspect='equal', axis='off'):
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, .8, .8])
    ax.set_aspect(aspect)
    if axis == 'off':
        ax.set_axis_off()
    return fig, ax


def model(model, name=None, color='k', dpi=100, ele=False, ele_label=False,
          surf_label=False, nodes_label=False, edges_label=False):
    """Plot the  model geometry

    """
    fig = plt.figure(name, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')
    ax.set_aspect('equal')

    draw.domain(model, ax, color=color)

    if ele is True:
        draw.elements(model, ax, color=color)

    if ele_label is True:
        draw.elements_label(model, ax)

    if surf_label is True:
        draw.surface_label(model, ax)

    if nodes_label is True:
        draw.nodes_label(model, ax)

    if edges_label is True:
        draw.edges_label(model, ax)

    return None


def model_deformed(model, U, magf=1, ele=False, name=None, color='Tomato',
                   dpi=100, ax=None):
    """Plot deformed model

    """
    if ax is None:
        fig = plt.figure(name, dpi=dpi)
        ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')

    if ele is True:
        draw.elements(model, ax, color='SteelBlue')
        draw.deformed_elements(model, U, ax, magf=magf, color=color)

    draw.domain(model, ax, color='SteelBlue')
    # draw.deformed_domain(model, U, ax, magf=magf, color=color)


def stresses(model, SIG, ftr=1, s11=False, s12=False, s22=False, spmax=False,
             spmin=False, dpi=100, name=None, lev=20, vmin=None, vmax=None,
             cbar_orientation='vertical', title=''):
    """Plot stress with nodal stresses

    """
    fig, ax = initiate()
    ax.set_xlabel(r'x')
    ax.set_ylabel(r'y')

    if s11 is True:
        ax.set_title(title)
        draw.tricontourf(model, SIG[:, 0]/ftr, ax, 'spring', lev=lev,
                         vmin=vmin, vmax=vmax, cbar_orientation=cbar_orientation)

    if s12 is True:
        ax.set_title(title)
        draw.tricontourf(model, SIG[:, 2]/ftr, ax, 'cool', lev=lev,
                         vmin=vmin, vmax=vmax, cbar_orientation=cbar_orientation)

    if s22 is True:
        ax.set_title(title)
        draw.tricontourf(model, SIG[:, 1]/ftr, ax, 'autumn', lev=lev,
                         vmin=vmin, vmax=vmax, cbar_orientation=cbar_orientation)

    if spmax is True:
        spmx = stress.principal_max(SIG[:, 0], SIG[:, 1], SIG[:, 2])
        ax.set_title(title)
        draw.tricontourf(model, spmx/ftr, ax, 'plasma', lev=lev,
                         vmin=vmin, vmax=vmax, cbar_orientation=cbar_orientation)

    if spmin is True:
        spmn = stress.principal_min(SIG[:, 0], SIG[:, 1], SIG[:, 2])
        ax.set_title(title)
        draw.tricontourf(model, spmn/ftr, ax, 'viridis', lev=lev,
                         vmin=vmin, vmax=vmax, cbar_orientation=cbar_orientation)


def model_deformed_dyn(model, U, ax, magf=1, ele=False, name=None,
                       color='Tomato',
                       dpi=100):
    """Plot deformed model

    """
    if ele is True:
        im = draw.deformed_elements_dyn(model, U, ax, magf=magf, color=color)
    else:
        im = draw.deformed_domain_dyn(model, U, ax, magf=magf,
                                      color=color)
    return im


def anime(frames, fig, t_int, interval=100):
    """Plot animation with images frames

    """
    ani = animation.ArtistAnimation(fig, frames, interval=interval,
                                    blit=True)
    return ani


def stresses_dyn(model, SIG, ax, ftr=1, s11=False, s12=False, s22=False,
                 spmax=False, spmin=False, dpi=100, name=None,
                 lev=20, vmin=None, vmax=None):
    """Plot stress with nodal stresses

    Return:
    im = list with matplotlib Artist

    """
    if s11 is True:
        s_range = [np.amin(s11), np.amax(s11)]
        ax.set_title(r'Temperature $^{\circ} C$')
        im = draw.tricontourf_dyn(model, SIG[:, 0]/ftr, ax, 'hot', lev=lev)

    if s12 is True:
        s_range = [np.amin(s12), np.amax(s12)]
        ax.set_title(r'Stress 12 ('+str(ftr)+' Pa)')
        im = draw.tricontourf_dyn(model, SIG[:, 2]/ftr, ax, 'cool', lev=lev)

    if s22 is True:
        s_range = [np.amin(s22), np.amax(s22)]
        ax.set_title(r'Stress 22 ('+str(ftr)+' Pa)')
        im = draw.tricontourf_dyn(model, SIG[:, 1]/ftr, ax, 'autumn', lev=lev)

    if spmax is True:
        spmx = stress.principal_max(SIG[:, 0], SIG[:, 1], SIG[:, 2])
        s_range = [np.amin(spmx), np.amax(spmx)]
        ax.set_title(r'Stress Principal Max ('+str(ftr)+' Pa)')
        im = draw.tricontourf_dyn(model, spmx/ftr, ax,
                                  'plasma', lev=lev, vmin=vmin, vmax=vmax)

    if spmin is True:
        spmn = stress.principal_min(SIG[:, 0], SIG[:, 1], SIG[:, 2])
        s_range = [np.amin(spmn), np.amax(spmn)]
        ax.set_title(r'Stress Principal Min ('+str(ftr)+' Pa)')
        im = draw.tricontourf_dyn(model, spmn/ftr, ax, 'viridis', lev=lev)

    return im, s_range


def stress_animation(SIG, model, t_int, dt, name="Stresses.gif", brate=500,
                     vmin=None, vmax=None, interval=100, ftr=1, lev=20,
                     show_plot=False,
                     **sig_plt):
    """Plot an animation gif for the stresses

    """
    N = int(t_int/dt)+1

    frm, srange = [], []

    fig, ax = initiate()

    for n in range(N):
        t = n*dt
        im, val_range = stresses_dyn(model, SIG[:, :, n], ax, **sig_plt,
                                     ftr=ftr, lev=lev)
        te = ax.text(0, 1, "Time (h): "+str(round(t/(60*60), 2)),
                     ha='left', va='top',
                     transform=ax.transAxes)
        frm.append(im + [te])
        srange.append(val_range)  # srange = [max, min]

    srange = np.array(srange)
    print('Min and Max: ', np.amin(srange), np.amax(srange))

    if 'spmax' in sig_plt:
        cmap_color = 'plasma'
    if 'spmin' in sig_plt:
        cmap_color = 'viridis'
    if 's11' in sig_plt:
        cmap_color = 'hot'
    else:
        cmap_color = 'plasma'

    # Change the colorbar range
    sm = plt.cm.ScalarMappable(cmap=cmap_color,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = plt.colorbar(sm)
    cbar.set_label(r'Temperature $^{\circ} C$')

    ani = anime(frm, fig, t_int, interval=interval)
    ani.save(name, writer='imagemagick', bitrate=brate)
    if show_plot is True:
        plt.show(block=False)


def displ_animation(U, model, t_int, dt, magf=1, name='displacement.gif',
                    brate=250, interval=100, show_plot=False):
    """Plot an animation for the displacement

    """
    N = int(t_int/dt)+1

    fig, ax = initiate()

    frm = []

    for n in range(N):
        t = n*dt
        im = model_deformed_dyn(model, U[:, n], ax, ele=True,
                                magf=magf)
        te = ax.text(.5, 1, "Time (h): "+str(round(t/(60*60), 2)), ha='center',
                     va='top', transform=ax.transAxes)
        frm.append([im, te])

    ani = anime(frm, fig, t_int, interval=interval)
    ani.save(name, writer='imagemagick', bitrate=brate)
    plt.show()


def stress_through_time(model, sig, node, t_int, dt, time_scale='day',
                        ax=None, ylabel='',
                        label='...', linestyle=None, marker=None, title=None):
    """Plott solution at a specific node through time
    
    """
    print('Plotting Stress through time at node {} ...'.format(node), end='')
    if ax is None:
        fig, ax = plt.subplots()

    if time_scale == 'day':
        time_factor = 60*60*24
    elif time_scale == 'hour':
        time_factor = 60*60
    else:
        print('Time scale should be hour or day!')

    # t_int in seconds
    number_steps = int(t_int/dt)
    t = np.linspace(0, t_int, number_steps+1)

    ax.plot(t/time_factor, sig/1e6, label=label,
            linestyle=linestyle, marker=marker, mfc='none', mew=.5)
    ax.set_xlabel('Time ({})'.format(time_scale))
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    print('Done')


def stress_along_y_at_time(model, sig, time, t_int, dt, time_scale='day',
                           ax=None, x=0, label=None, marker=None, ylabel=None,
                           ftr=1, linestyle=None):
    print('Plotting solution at time {} through the y-axis ...'.format(time), end='')
    if ax is None:
        fig, ax = plt.subplots()

    if time_scale == 'day':
        # (t_int/dt) is the number of steps
        # (t_int/60*60*24) is the time interval in days
        time_index = int((t_int/dt)/(t_int/(60*60*24))*time)
    elif time_scale == 'hour':
        time_index = int((t_int/dt)/(t_int/(60*60))*time)
    else:
        print('Time scale should be hour or day!')

    nodes = np.where(np.round(model.XYZ[:, 0], 3) == x)[0]
    y = model.XYZ[nodes, 1]

    data = np.array(list(zip(y, sig[nodes, time_index])))

    sorted_data = data[np.argsort(data[:, 0])]

    # import os
    # np.savetxt('gisele_solution_in_y.txt',
    #            sorted_data,
    #            fmt='%.4f', newline=os.linesep)

    ax.plot(sorted_data[:, 1]/ftr, sorted_data[:, 0], label=label,
            marker=marker,
            linestyle=linestyle, mfc='none', mew=.5)
    ax.set_ylabel(r'$y (m)$')
    ax.set_xlabel(ylabel)
    ax.legend()
    plt.tight_layout()
    print('Done')
