import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import networkx as nx


def elements_label(model, ax):

    for e in range(len(model.CONN)):
        x_element = (model.XYZ[model.CONN[e, 0], 0] +
                     model.XYZ[model.CONN[e, 1], 0] +
                     model.XYZ[model.CONN[e, 2], 0] +
                     model.XYZ[model.CONN[e, 3], 0])/4.

        y_element = (model.XYZ[model.CONN[e, 0], 1] +
                     model.XYZ[model.CONN[e, 1], 1] +
                     model.XYZ[model.CONN[e, 2], 1] +
                     model.XYZ[model.CONN[e, 3], 1])/4.

        ax.annotate(str(e), (x_element, y_element), size=9,
                    color='r')


def surface_label(model, ax):

    c = model.XYZ

    X, Y = c[:, 0], c[:, 1]

    G2 = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G2.add_node(i, posxy=(X[i], Y[i]))

    # adjust the node numbering order of an element
    if model.gmsh == 1.0:
        temp = np.copy(model.CONN[:, 2])
        model.CONN[:, 2] = model.CONN[:, 3]
        model.CONN[:, 3] = temp
        model.gmsh += 1

    i = 0
    for surface, lpTag in model.physical_surf.items():
        xm = 0.0
        ym = 0.0
        for node in model.line_loop[lpTag]:
            G2.add_edge(model.line[node][0],
                        model.line[node][1])
            xm += (model.XYZ[model.line[node][0], 0] +
                   model.XYZ[model.line[node][1], 0])
            ym += (model.XYZ[model.line[node][0], 1] +
                   model.XYZ[model.line[node][1], 1])

        xs, ys = xm/(2*len(model.line_loop[lpTag])), ym/(2*len(model.line_loop[
                                                                 lpTag]))
        sf = model.physical_surf[surface]
        ax.annotate(str(sf), (xs, ys), size=9, color='g')
        i += 1


def edges_label(model, ax):
    c = model.XYZ

    X, Y = c[:, 0], c[:, 1]

    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    bound_middle = {}
    iant = model.nodes_in_bound_line[0, 0]

    cont = 0
    for i, e1, e2 in model.nodes_in_bound_line:
        if i == iant:
            cont += 1
            bound_middle[i] = cont
        else:
            cont = 1
        iant = i

    edge_labels = {}
    cont = 0
    for i, e1, e2 in model.nodes_in_bound_line:
        cont += 1
        if cont == int(bound_middle[i]/2.0):
            edge_labels[e1, e2] = str(i)
        if cont == bound_middle[i]:
            cont = 0

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx_edge_labels(G, positions, edge_labels, label_pos=0.5,
                                 font_size=9, font_color='b', ax=ax)


def nodes_label(model, ax):
    c = model.XYZ

    X, Y = c[:, 0], c[:, 1]

    G = nx.Graph()

    label = {}
    for i in range(len(X)):
        label[i] = i
        G.add_node(i, posxy=(X[i], Y[i]))

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx_nodes(G, positions, node_color='w', node_size=140,
                           node_shape='s', ax=ax)
    nx.draw_networkx_labels(G, positions, label, font_size=9, ax=ax)


def domain(model, ax, color):
    """Draw domain region

    """
    c = model.XYZ

    X, Y = c[:, 0], c[:, 1]

    G = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G.add_node(i, posxy=(X[i], Y[i]))

    for plTag, lineTag in model.physical_line.items():
        lineNodes = model.line[lineTag]
        G.add_edge(lineNodes[0], lineNodes[1])

    positions = nx.get_node_attributes(G, 'posxy')

    nx.draw_networkx_edges(G, positions, edge_color=color,
                           font_size=0, width=1, origin='lower', ax=ax)


def elements(model, ax, color):

    c = model.XYZ

    X, Y = c[:, 0], c[:, 1]

    G2 = nx.Graph()

    label = []
    for i in range(len(X)):
        label.append(i)
        G2.add_node(i, posxy=(X[i], Y[i]))

    if model.gmsh == 1.0:
        temp = np.copy(model.CONN[:, 2])
        model.CONN[:, 2] = model.CONN[:, 3]
        model.CONN[:, 3] = temp
        model.gmsh += 1

    for i in range(len(model.CONN)):
        G2.add_cycle([model.CONN[i, 0],
                     model.CONN[i, 1],
                     model.CONN[i, 3],
                     model.CONN[i, 2]], )

    edge_line_nodes = {}
    for i in range(len(model.nodes_in_bound_line[:, 0])):
        edge_line_nodes[(model.nodes_in_bound_line[i, 1], model.nodes_in_bound_line[i,
                                                                        2])] \
            = model.nodes_in_bound_line[i, 0]

    positions = nx.get_node_attributes(G2, 'posxy')

    nx.draw_networkx(G2, positions, node_size=0, edge_color=color,
                     font_size=0,  width=1)


def deformed_elements(model, U, ax, magf, color):
    """Draw deformed elements

    """
    c = model.XYZ

    dX, dY = c[:, 0] + U[::2]*magf, c[:, 1] + U[1::2]*magf

    G2 = nx.Graph()

    if model.gmsh == 1.0:
        temp = np.copy(model.CONN[:, 2])
        model.CONN[:, 2] = model.CONN[:, 3]
        model.CONN[:, 3] = temp
        model.gmsh += 1.0

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(model.CONN)):
        G2.add_cycle([model.CONN[i, 0],
                     model.CONN[i, 1],
                     model.CONN[i, 3],
                     model.CONN[i, 2]], )

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    nx.draw_networkx(G2, positions2, node_size=0, edge_color=color,
                     font_size=0, width=1, ax=ax)


def deformed_domain(model, U, ax, magf, color):
    """Draw deformed domain

    """
    c = model.XYZ
    bn = model.nodes_in_bound_line

    adX = U[::2]
    adY = U[1::2]

    dX, dY = (c[bn[:, 1], 0] + adX[bn[:, 1]]*magf,
              c[bn[:, 1], 1] + adY[bn[:, 1]]*magf)
    G2 = nx.Graph()

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(bn[:, 0]) - 1):
        G2.add_edge(i, i+1)

    G2.add_edge(len(bn[:, 0]) - 1, 0)

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    nx.draw_networkx_edges(G2, positions2, edge_color=color,
                     font_size=0, width=1, ax=ax)


def draw_bc_dirichlet(displacement, model, name, dpi):

    plt.figure(name, dpi=dpi)

    h = model.AvgLength


    for line in displacement(1,1).keys():
        d= displacement(1,1)[line]
        if d[0] == 0.0 and d[1]== 0.0:
            for l, n1, n2 in model.nodes_in_bound_line:
                if line[1] == l:
                    x1 = model.XYZ[n1, 0]
                    y1 = model.XYZ[n1, 1]
                    plt.annotate('', xy=(x1, y1), xycoords='data',
                                 xytext=(x1-h,y1-h), textcoords='data',
                                 arrowprops=dict(facecolor='black', width=0.2,
                                                 headwidth=0.2))

                    x2 = model.XYZ[n2, 0]
                    y2 = model.XYZ[n2, 1]
                    plt.annotate('', xy=(x2, y2), xycoords='data',
                                 xytext=(x2-h,y2-h), textcoords='data',
                                 arrowprops=dict(facecolor='black', width=0.2,
                                                 headwidth=0.2))

    plt.axes().set_aspect('equal')

    plt.axes().autoscale_view(True, True, True)
    plt.margins(y=0.1, x=0.1, tight=False)
    plt.draw()


def draw_bc_newmann(traction, model, name, dpi):

    plt.figure(name, dpi=dpi)

    h = model.AvgLength


    for line in traction(1,1).keys():
        t = traction(1,1)[line]
        if t[0] != 0.0 or t[1] != 0.0:
            for l, n1, n2 in model.nodes_in_bound_line:
                if line[1] == l:
                    x1 = model.XYZ[n1, 0]
                    y1 = model.XYZ[n1, 1]
                    t1 = traction(x1, y1)[line]
                    t_r1 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    w=5000.0
                    plt.annotate('', xy=(x1, y1), xycoords='data',
                                 xytext=(x1 - t1[0]/w, y1 - t1[1]/w),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=4, shrink=0))

                    t_r2 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    x2 = model.XYZ[n2, 0]
                    y2 = model.XYZ[n2, 1]
                    t2 = traction(x2, y2)[line]
                    plt.annotate('', xy=(x2, y2), xycoords='data',
                                 xytext=(x2 - t2[0]/w, y2 - t2[1]/w),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=4, shrink=0))
    plt.axes().set_aspect('equal')

    plt.axes().autoscale_view(True, True, True)
    plt.margins(y=0.1, x=0.1, tight=False)
    plt.draw()


def draw_bc_neumann_value(traction, model, name, dpi):

    plt.figure(name, dpi=dpi)

    h = model.AvgLength

    for line in traction(1, 1).keys():
        t = traction(1, 1)[line]
        if t[0] != 0.0 or t[1] != 0.0:
            for l, n1, n2 in model.nodes_in_bound_line:
                if line == l:
                    x1 = model.XYZ[n1, 0]
                    y1 = model.XYZ[n1, 1]
                    t1 = traction(x1, y1)[line]
                    t_r1 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    plt.annotate(str(t_r1), xy=(x1, y1), xycoords='data',
                                 xytext=(x1 - t1[0], y1 - t1[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))

                    t_r2 = np.sqrt(t1[0]**2.+t1[1]**2.)
                    x2 = model.XYZ[n2, 0]
                    y2 = model.XYZ[n2, 1]
                    t2 = traction(x2, y2)[line]
                    plt.annotate(str(t_r2), xy=(x2, y2), xycoords='data',
                                 xytext=(x2 - t2[0], y2 - t2[1]),
                                 textcoords='data', size=8,
                                 verticalalignment='top',
                                 arrowprops=dict(facecolor='black', width=0,
                                                 headwidth=5, shrink=0.1))


def tricontourf(model, sig, ax, cmap, lev, vmin=None, vmax=None,
                cbar_orientation='vertical'):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    """
    c = model.XYZ

    bn = model.nodes_in_bound_line

    xx, yy, zz = c[:, 0], c[:, 1],  sig

    triangles = []
    for n1, n2, n3, n4 in model.CONN:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    triangles = np.asarray(triangles)

    if vmin is None:
        vmin = np.amin(sig)
    if vmax is None:
        vmax = np.amax(sig)

    CS2 = ax.tricontourf(xx, yy, triangles, zz, lev,
                         origin='lower',
                         cmap=cmap, antialiased=True,
                         vmin=vmin, vmax=vmax)

    CS3 = ax.tricontour(xx, yy, triangles, zz, lev, colors='black')
    ax.clabel(CS3, fontsize=8, colors='k', fmt='%1.1f')

    # # Change the colorbar range
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = plt.colorbar(sm, orientation=cbar_orientation)



def tricontourf_dyn(model, sig, ax, cmap, lev, vmin=None, vmax=None):
    """Plot contour with the tricoutour function and the boundary line with
    the boundary node.

    Return:
    list with plot Artist

    """
    c = model.XYZ

    bn = model.nodes_in_bound_line

    xx, yy, zz = c[:, 0], c[:, 1],  sig

    ccx = np.append(c[bn[:, 1], 0], c[bn[0, 1], 0])
    ccy = np.append(c[bn[:, 1], 1], c[bn[0, 1], 1])

    triangles = []
    for n1, n2, n3, n4 in model.CONN:
        triangles.append([n1, n2, n3])
        triangles.append([n1, n3, n4])

    triangles = np.asarray(triangles)

    CS2 = ax.tricontourf(xx, yy, triangles, zz, lev,
                         origin='lower',
                         cmap=cmap, antialiased=True,
                         vmin=vmin, vmax=vmax)

    CS3 = ax.plot(ccx, ccy, '-k')

    return CS2.collections + CS3


def deformed_domain_dyn(model, U, ax, magf, color):
    """Draw deformed domain

    """
    c = model.XYZ

    bn = model.nodes_in_bound_line

    adX = U[::2]
    adY = U[1::2]

    dX, dY = (c[bn[:, 1], 0] + adX[bn[:, 1]]*magf,
              c[bn[:, 1], 1] + adY[bn[:, 1]]*magf)

    G2 = nx.Graph()

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(bn[:, 0]) - 1):
        G2.add_edge(i, i+1)

    G2.add_edge(len(bn[:, 0]) - 1, 0)

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    im = nx.draw_networkx_edges(G2, positions2, node_size=0, edge_color=color,
                                font_size=0, width=1, ax=ax)

    return im


def deformed_elements_dyn(model, U, ax, magf, color):
    """Draw deformed elements

    """
    c = model.XYZ

    dX, dY = c[:, 0] + U[::2]*magf, c[:, 1] + U[1::2]*magf

    G2 = nx.Graph()

    if model.gmsh == 1.0:
        temp = np.copy(model.CONN[:, 2])
        model.CONN[:, 2] = model.CONN[:, 3]
        model.CONN[:, 3] = temp
        model.gmsh += 1.0

    label2 = []
    for i in range(len(dX)):
        label2.append(i)
        G2.add_node(i, posxy2=(dX[i], dY[i]))

    for i in range(len(model.CONN)):
        G2.add_cycle([model.CONN[i, 0],
                     model.CONN[i, 1],
                     model.CONN[i, 3],
                     model.CONN[i, 2]], )

    positions2 = nx.get_node_attributes(G2, 'posxy2')

    im = nx.draw_networkx_edges(G2, positions2, edge_color=color,
                                font_size=0, width=1, ax=ax)

    return im
