"""Module for enriched quad element with 4 nodes - type 3 in gmsh

"""
import numpy as np
from .quad4 import Quad4


class Quad4Enr(Quad4):
    """Constructor of a 4-node quadrangle (TYPE 3) enriched element

    Uses methods of the standard Quad4 element.

    Args:
        eid: element index
        model: object with model parameters
        eps0: element inital strain array shape [3]

    Attributes:
        phi (numpy array): nodal value for the signed distance function
            of this element
        num_enr_nodes (int): number of enriched nodes of this element
        num_enr_dof (int): number of enriched degree`s of freedom in this
            element.
    """
    def __init__(self, eid, model, EPS0):
        # initialize Quad4 standard
        super().__init__(eid, model, EPS0)

    def stiffness_matrix(self, t=1):
        """Build the enriched element stiffness matrix

        Note:
            This method overwrites the Quad4 method

        """
        kuu = np.zeros((self.num_std_dof, self.num_std_dof))
        kaa = np.zeros((self.num_enr_dof, self.num_enr_dof))
        kua = np.zeros((self.num_std_dof, self.num_enr_dof))

        for w, gp in zip(self.gauss_quad.weights, self.gauss_quad.points):
            N, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            C = self.c_matrix(N, t)
            Bstd = self.standard_gradient_operator(dN_xi)
            Benr = self.enriched_gradient_operator(N, dN_xi)

            kuu += w*(Bstd.T @ C @ Bstd)*dJ
            kaa += w*(Benr.T @ C @ Benr)*dJ
            kua += w*(Bstd.T @ C @ Benr)*dJ

        k = np.block([[kuu, kua],
                      [kua.T, kaa]])
        return k * self.thickness

    def enriched_gradient_operator(self, N, dN_xi):
        """Build the enriched gradient operator

        Args:
            dN_xi: derivative of shape functions with respect to cartesian
                coordinates

        Returns:
            numpy array shape (3 x (num_enr_dof))

        Note:
            The order of this matrix is defined by the order in the
            element.dof.

            For instance, (example in the test_quad4enr)

                element[1].dof =  [2, 3, 10, 11, 8, 9, 4, 5,  # std
                                   14, 15, 16, 17,  # first zls
                                   20, 21, 22, 23, 24, 25, 26, 27]  # 2nd  zls

            which is an element enriched by two zero level set. In this case,
            the gradient operator, B, should follow this order.
            The attribute enriched dof for each level set keeps track of those
            dof:

                model.zerolevelset[0].enriched_dof[1] = [14, 15, 16, 17]
                model.zerolevelset[1].enriched_dof[1] = [20, 21, 22, 23,
                                                         24, 25, 26, 27]

            The, we need to know which nodes those dof are refering to in the
            local index. This are stored in element enriched node,

                element[1].enriched_nodes[0] = [1, 2]  # for 1st zls
                element[1].enriched_nodes[1] = [1, 2, 4, 5]  # for 2nd zls

        """
        # dof = [std dofs, dofs zls[0], dofs zls[1] ... ]
        # loop for each zero level set
        Benr_zls = {}   # Benr for each zerp level est
        for ind, zls in enumerate(self.zerolevelset):
            # signed distance for nodes in this element for this zls
            phi = zls.phi[self.conn]  # phi with local index

            Bk = {}         # Bk for k enriched nodes
            # enriched nodes [[nodes for the first level set], [for 2nd]]
            for n in self.enriched_nodes[ind]:
                # enriched_nodes is sorted
                # if conn = [1, 4, 7, 2] and n = 4 then j = 1
                j = self.global2local_index(n)  # local index
                psi = abs(N @ phi) - abs(phi[j])

                dpsi_x = np.sign(N @ phi)*(dN_xi[0, :] @ phi)
                dpsi_y = np.sign(N @ phi)*(dN_xi[1, :] @ phi)

                # store Bk using node index
                # but use local index to access phi, N, dN_xi
                Bk[n] = np.array([[dN_xi[0, j]*(psi) + N[j]*dpsi_x, 0],
                                  [0, dN_xi[1, j]*(psi) + N[j]*dpsi_y],
                                  [dN_xi[1, j]*(psi) + N[j]*dpsi_y,
                                   dN_xi[0, j]*(psi) + N[j]*dpsi_x]])

            # Arrange Benr based on the element.enriched_nodes
            Benr_zls[ind] = np.block([Bk[i]
                                      for i in self.enriched_nodes[ind]])

        # arrange Benr based on zls order
        Benr = np.block([Benr_zls[i]
                         for i in range(len(self.zerolevelset))])
        return Benr

    def load_body_vector(self, b_force=None, t=1):
        """Build the element vector due body forces b_force

        """
        gauss_points = self.xez / np.sqrt(3.0)

        pb_std = np.zeros(self.num_std_dof)
        pb_enr = np.zeros(self.num_enr_dof)

        if b_force is not None:
            for gp in gauss_points:
                N, dN_ei = self.shape_function(xez=gp)
                dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

                x1, x2 = self.mapping(N, self.xyz)

                pb_std[0] += N[0]*b_force(x1, x2, t)[0]*dJ
                pb_std[1] += N[0]*b_force(x1, x2, t)[1]*dJ
                pb_std[2] += N[1]*b_force(x1, x2, t)[0]*dJ
                pb_std[3] += N[1]*b_force(x1, x2, t)[1]*dJ
                pb_std[4] += N[2]*b_force(x1, x2, t)[0]*dJ
                pb_std[5] += N[2]*b_force(x1, x2, t)[1]*dJ
                pb_std[6] += N[3]*b_force(x1, x2, t)[0]*dJ
                pb_std[7] += N[3]*b_force(x1, x2, t)[1]*dJ

        pb = np.block([pb_std, pb_enr])
        return pb * self.thickness

    def load_strain_vector(self, t=1):
        """Build the element vector due initial strain

        """
        pe_std = np.zeros(self.num_std_dof)
        pe_enr = np.zeros(self.num_enr_dof)

        for w, gp in zip(self.gauss_quad.weights, self.gauss_quad.points):
            N, dN_ei = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_ei)

            C = self.c_matrix(N, t)

            B = np.array([
                [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                 dN_xi[0, 3], 0],
                [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                 dN_xi[1, 3]],
                [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                 dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            pe_std += w*(B.T @ C @ self.eps0)*dJ

        pe = np.block([pe_std, pe_enr])
        return pe * self.thickness

    def load_traction_vector(self, traction_bc=None, t=1):
        """Build element load vector due traction_bction boundary condition

        Performs the line integral on boundary with traction vector assigned.

        1. First loop over the boundary lines tag which are the keys of the
           traction_bc dictionary.
        2. Then loop over elements at boundary lines and which side of the
           element is at the boundary line.

        Note:
            This method overwrites the Quad4 method

        Args:
            traction_bc (dict): dictionary with boundary conditions with the
                format: traction_bc = {line_tag: [vector_x, vector_y]}.
        """
        gp = np.array([
            [[-1.0/np.sqrt(3), -1.0],
             [1.0/np.sqrt(3), -1.0]],
            [[1.0, -1.0/np.sqrt(3)],
             [1.0, 1.0/np.sqrt(3)]],
            [[-1.0/np.sqrt(3), 1.0],
             [1.0/np.sqrt(3), 1.0]],
            [[-1.0, -1.0/np.sqrt(3)],
             [-1.0, 1/np.sqrt(3)]]])

        pt_std = np.zeros(self.num_std_dof)
        pt_enr = np.zeros(self.num_enr_dof)

        if traction_bc is not None:
            # loop for specified boundary conditions
            for key in traction_bc(1, 1).keys():
                line = key[1]

                for ele_boundary_line, ele_side in zip(self.at_boundary_line,
                                                       self.side_at_boundary):
                    # Check if this element is at the line with traction
                    if line == ele_boundary_line:
                        # perform the integral with GQ
                        for w in range(2):
                            N, dN_ei = self.shape_function(xez=gp[ele_side, w])
                            _, _, arch_length = self.jacobian(self.xyz, dN_ei)

                            dL = arch_length[ele_side]
                            x1, x2 = self.mapping(N, self.xyz)

                            # Nstd matrix with shape function shape (2, 8)
                            N_ = []
                            for j in range(self.num_std_nodes):
                                N_.append(np.array([[N[j], 0],
                                                    [0, N[j]]]))
                            Nstd = np.block([N_[j]
                                             for j
                                             in range(self.num_std_nodes)])

                            # traction_bc(x1, x2, t)[key] is a numpy
                            # array shape (2,)
                            pt_std += Nstd.T @ traction_bc(x1, x2, t)[key] * dL
                    else:
                        # Catch element that is not at boundary
                        continue

        pt = np.block([pt_std, pt_enr])
        return pt
