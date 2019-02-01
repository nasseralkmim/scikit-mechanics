"""plot enriched nodes"""


def plot_enriched_nodes(model, ax):
    """Plot enriched nodes

    Parameters
    ----------
    model : object
        model object with model.xfem object attribute

    """
    for nid in model.xfem.enr_nodes:
        ax.plot(model.nodes[nid][0], model.nodes[nid][1], marker='o',
                color='r', mfc='none')
