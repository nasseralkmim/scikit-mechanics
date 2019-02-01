"""Save output fields in gmsh file and text files for incremental analysis"""
from ..postprocess.dof2node import dof2node
from ..postprocess.writeoutput import write_output
from ..meshplotlib.gmshio.gmshio import write_field
from ..postprocess.stressrecovery import extrapolate_gp_smoothed


def save_output(model, u, int_var, increment, start, lmbda,
                element_out=None, node_out=None):
    """Save output to .msh file"""
    displ_dic = dof2node(u, model)
    write_field(displ_dic, model.mesh.name,
                'Displacement', 2, lmbda, increment, start)

    if node_out is not None:
        write_output(lmbda, displ_dic[node_out],
                     f'{model.mesh.name}_displ_node{node_out}', start)

    # smoothed (average) extrapolated stresses to nodes
    sig_x = {(eid, gp): int_var['sig'][(eid, gp)][0]
             for eid in model.elements.keys()
             for gp in range(model.num_quad_points[eid] * 2)}
    sig_y = {(eid, gp): int_var['sig'][(eid, gp)][1]
             for eid in model.elements.keys()
             for gp in range(model.num_quad_points[eid] * 2)}
    sig_z = {(eid, gp): int_var['sig'][(eid, gp)][3]
             for eid in model.elements.keys()
             for gp in range(model.num_quad_points[eid] * 2)}
    sig_xy = {(eid, gp): int_var['sig'][(eid, gp)][2]
              for eid in model.elements.keys()
              for gp in range(model.num_quad_points[eid] * 2)}
    sig_x_dic = extrapolate_gp_smoothed(model, sig_x)
    sig_y_dic = extrapolate_gp_smoothed(model, sig_y)
    sig_z_dic = extrapolate_gp_smoothed(model, sig_z)
    sig_xy_dic = extrapolate_gp_smoothed(model, sig_xy)

    write_field(sig_x_dic, model.mesh.name,
                'Sigma x', 1, lmbda, increment, start)
    # write_field(sig_y_dic, model.mesh.name,
    #             'Sigma y', 1, lmbda, increment, start)
    # write_field(sig_z_dic, model.mesh.name,
    #             'Sigma z', 1, lmbda, increment, start)
    # write_field(sig_xy_dic, model.mesh.name,
    #             'Sigma xy', 1, lmbda, increment, start)

    if model.micromodel is None:
        sig_vm_dic = extrapolate_gp_smoothed(model, int_var['q'])
        write_field(sig_vm_dic, model.mesh.name,
                    'Von Mises', 1, lmbda, increment, start)
        # element average of cummulative plastic strain
        eps_bar_p_dic = extrapolate_gp_smoothed(model, int_var['eps_bar_p'])
        write_field(eps_bar_p_dic, model.mesh.name,
                    'Cummulative plastic strain', 1, lmbda, increment, start)

    if element_out is not None:
        if model.micromodel is None:
            write_output(lmbda, int_var['q'][(element_out, 0)],
                         f'{model.mesh.name}_mises_gp1_ele{element_out}', start)
            write_output(lmbda, int_var['q'][(element_out, 1)],
                         f'{model.mesh.name}_mises_gp2_ele{element_out}', start)
            write_output(lmbda, int_var['q'][(element_out, 2)],
                         f'{model.mesh.name}_mises_gp3_ele{element_out}', start)
            write_output(lmbda, int_var['q'][(element_out, 3)],
                         f'{model.mesh.name}_mises_gp4_ele{element_out}', start)

            write_output(lmbda, int_var['eps_bar_p'][(element_out, 0)],
                         f'{model.mesh.name}_peeq_gp1_ele{element_out}', start)
            write_output(lmbda, int_var['eps_bar_p'][(element_out, 1)],
                         f'{model.mesh.name}_peeq_gp2_ele{element_out}', start)
            write_output(lmbda, int_var['eps_bar_p'][(element_out, 2)],
                         f'{model.mesh.name}_peeq_gp3_ele{element_out}', start)
            write_output(lmbda, int_var['eps_bar_p'][(element_out, 3)],
                         f'{model.mesh.name}_peeq_gp4_ele{element_out}', start)

        write_output(lmbda, int_var['sig'][(element_out, 0)],
                     f'{model.mesh.name}_sig_gp1_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 1)],
                     f'{model.mesh.name}_sig_gp2_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 2)],
                     f'{model.mesh.name}_sig_gp3_ele{element_out}', start)
        write_output(lmbda, int_var['sig'][(element_out, 3)],
                     f'{model.mesh.name}_sig_gp4_ele{element_out}', start)

        write_output(lmbda, int_var['eps'][(element_out, 0)],
                     f'{model.mesh.name}_eps_gp1_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 1)],
                     f'{model.mesh.name}_eps_gp2_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 2)],
                     f'{model.mesh.name}_eps_gp3_ele{element_out}', start)
        write_output(lmbda, int_var['eps'][(element_out, 3)],
                     f'{model.mesh.name}_eps_gp4_ele{element_out}', start)

    if model.micromodel is not None:
        # update micro displacement after macro NR converged
        u_mic = model.micromodel.update_displ(element_out)
        displ_dic = dof2node(
            u_mic, model.micromodel)
        write_field(displ_dic, model.micromodel.mesh.name,
                    'Displacement', 2, lmbda, increment, start)

        # plot stress xx component
        sig_x = {
            (eid, gp): int_var['sig_mic'][(element_out, 3)][(eid, gp)][0]
            for eid in model.micromodel.elements.keys()
            for gp in range(model.micromodel.num_quad_points[eid] * 2)}
        sig_x_dic = extrapolate_gp_smoothed(
            model.micromodel, sig_x)
        write_field(
            sig_x_dic, model.micromodel.mesh.name,
            f'Sigma x Element {element_out} GP 4', 1, lmbda, increment, start)

        # plot equivalent plastic strain for microscale
        eps_bar_p= {
            (eid, gp): int_var['eps_bar_p_mic'][(element_out, 3)][(eid, gp)]
            for eid in model.micromodel.elements.keys()
            for gp in range(model.micromodel.num_quad_points[eid] * 2)}
        eps_bar_p_dic = extrapolate_gp_smoothed(
            model.micromodel, eps_bar_p)
        write_field(
            eps_bar_p_dic, model.micromodel.mesh.name,
            f'Eps bar p Element {element_out} GP 4', 1, lmbda, increment, start)

    return None
