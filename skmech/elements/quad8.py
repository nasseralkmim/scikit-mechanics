"""8 node quadrangle gmsh type 16
"""
import numpy as np
from ..element import Element
from .. import quadrature


class Quad8(Element):
    """Constructor of a 4-node quadrangle (TYPE 3) element

    Attributes:
        at_boundary_line (list): boundary line tag in which this element is
        side_at_boundary (list): side of the element at the boundary line.
        nodes_in_ele_bound (dict): element side and nodes at this side
        E (float or callable or list): elastic modulus of this element, if
            float is constant for element (which is in a surface), if
            callable is a function that will be evaluated when integrating,
            if list is a float for each node in element defined by the sign of
            the phi array (signed distance from zero level set).
        nu (floaf): Poisson's ration of this element
        eps0 (numpy array): shape (3,) initial strain vector (s11, s22, s12)
        XEZ (numpy array): shape (4, 2): coordinates of nodes in isoparametric
            domain
        enriched_nodes (list): nodes that are enriched, if None results in
            empty list

    Args:
        eid: element index
        model: object with model parameters
        EPS0: element inital strain array shape [3]

    Note:
        Element material is assigned as a constant for the element, as a
        function of space or as discrete nodal values. In the last case, the
        zero level set is used to define different regions. Negative 1 is a
        negative and positive 1 is another. The nodal values are interpolated
        in the integrals using the shape functions.

    """
    def __init__(self, eid, model, EPS0):
        super().__init__(eid, model)

        # gauss points
        self.gauss_quad = quadrature.Quadrilateral(self.num_quad_points)
        self.num_gp = len(self.gauss_quad.points)

        try:
            if model.xfem:
                # E = [E for nodes in element]
                # In this case self.E will be defined for each node in element
                # based on level set phi. If level set has negative sign, then
                # it is a surface, a positive indicates another
                # -1 -> zls.region['reinforcement']
                # +1 -> zls.region['matrix']

                # zls.material.E = {-1: Evalue, 1: value}
                # initialize material with matrix value
                self.E = [model.E_matrix]*self.num_std_nodes
                self.nu = [model.nu_matrix]*self.num_std_nodes
                # loop over element zero level set
                for zls in model.zerolevelset:
                    # loop over element nodes with negative phi values
                    for j in np.where(zls.phi[self.conn] < 0)[0]:
                        self.E[j] = zls.material.E[-1]
                        self.nu[j] = zls.material.nu[-1]
            else:
                self.E = model.material.E[self.surf]
                self.nu = model.material.nu[self.surf]
        except AttributeError:
            print('E and nu must be defined for all surfaces! (Default used)')
        except KeyError:
            print('Surface ', self.surf,
                  ' with no material assigned! (Default used)')

        # create an initial strain due non mechanical effect
        if EPS0 is None:
            self.eps0 = np.zeros(3)
        else:
            self.eps0 = EPS0[eid]

        # check if its a boundary element
        if eid in model.bound_ele[:, 0]:
            # index where bound_ele refers to this element
            index = np.where(model.bound_ele[:, 0] == eid)[0]
            # side of the element at the boundary
            self.side_at_boundary = model.bound_ele[index, 1]
            # boundary line where the element side share interface
            self.at_boundary_line = model.bound_ele[index, 2]
        else:
            self.side_at_boundary = []
            self.at_boundary_line = []

        # use this in traction boundary condition
        # element_side, node 1 and node 2 in local tag
        # side 0 is bottom
        self.local_nodes_in_side = {0: [0, 1],
                                    1: [1, 2],
                                    2: [2, 3],
                                    3: [3, 1]}

        # TODO: make this better
        # 1. go over each element side and the correspondent boundary line
        # 2. find the nodes in the same line using model.nodes_in_bound_line
        # 3. loop in model.nodes_in_bound_line
        # 4. check if the node is in this element
        self.nodes_in_ele_bound = {}
        for line, ele_side in zip(self.at_boundary_line,
                                  self.side_at_boundary):
            n_ = model.nodes_in_bound_line
            for l, n1, n2 in n_[np.where(n_[:, 0] == line)]:
                if n1 in self.conn and n2 in self.conn:
                    self.nodes_in_ele_bound[ele_side] = [ele_side, n1, n2]

    @staticmethod
    def shape_function(self, xez):
        """Create the basis function and evaluate them at xez coordinates

        Arguments
        ---------
            xez (array): position in the isoparametric coordinate xi, eta, zeta

        Returns
        -------
            N (array): shape functions

        Note:
            The shape function Ni(xi, eta) are given for corner nodes couting
            from lower left corner. The nodes cooridnate are (xi_i, eta_i)
            where i = 0, 1, 2, 3, 4, 5, 6, 7.

                3-------6--------2
                |       ^eta     |
                |       |        |
                7       |--->xi  5
                |                |
                |                |
                0-------4--------1

            Corner nodes (0, 1, 2, 3)

                Ni = 1/4*(1 + xi_i * xi)*(1 + eta_i * eta)*(
                    xi * xi_i + eta * eta_i - 1)

            Mid nodes for xi_i = 0, (7, 5)

                Ni = 1/2*(1 - xi**2)*(1 + eta * eta_i)

            for eta_i = 0, Nodes (4, 6):

                Ni = 1/2*(1 - eta**2)*(1 + xi * xi_i)
        """
        # variables in the natural (iso-parametric) domain
        xi, eta = xez[0], xez[1]

        # Nodal coordinates in the natural domain (isoparametric coordinates)
        # This defines the local node numbering, following gmsh convention
        xez = np.array([[-1.0, -1.0],
                        [1.0, -1.0],
                        [1.0, 1.0],
                        [-1.0, 1.0],
                        [0., -1.],
                        [1., 0.],
                        [0., 1.],
                        [-1., 0.]])
        xi_i, eta_i = xez[:, 0], xez[:, 1]

        # Basis functions
        # N = [ N_0 N_1 N_2 N_3 N_4 N_5 N_6 N_7]
        self.N = np.array([
            1/4*(1 + xi_i[0]*xi)*(1 + eta_i[0]*eta),
            1/4*(1 + xi_i[1]*xi)*(1 + eta_i[1]*eta),
            1/4*(1 + xi_i[2]*xi)*(1 + eta_i[2]*eta),
            1/4*(1 + xi_i[3]*xi)*(1 + eta_i[3]*eta),
            1/2*(1 - xi**2)*(1 + eta * eta_i[4]),
            1/2*(1 - eta**2)*(1 + xi * xi_i[5]),
            1/2*(1 - xi**2)*(1 + eta * eta_i[6]),
            1/2*(1 - eta**2)*(1 + xi * xi_i[7])])

        # Derivative of the shape functions
        # dN = [ dN1_xi dN2_xi ...
        #         dN1_eta dN2_eta ... ]
        self.dN_Xi = np.array([
            [1/4*(xi_i[0])*(1 + eta_i[0]*eta),
             1/4*(xi_i[1])*(1 + eta_i[1]*eta),
             1/4*(xi_i[2])*(1 + eta_i[2]*eta),
             1/4*(xi_i[3])*(1 + eta_i[3]*eta),
             1/2*(1 - xi**2)*(1 + eta * eta_i[4]),
             1/2*(1 - eta**2)*(1 + xi * xi_i[5]),
             1/2*(1 - xi**2)*(1 + eta * eta_i[6]),
             1/2*(1 - eta**2)*(1 + xi * xi_i[7])],
            []
        ])

        return self.N, self.dN_Xi

    def mapping(self, N, xyz):
        """maps from cartesian to isoparametric.

        """
        x1, x2 = N @ xyz
        return x1, x2

    def global2local_index(self, n):
        """Returns local index from global node index

        Arguments
        ---------
        n (int): global index value for node

        Returns
        -------
        ind (int) local index

        Example
        -------
        >>> conn = [4, 5, 6, 7]
        >>> global2local_index(4)
        0
        >>> global2local_index(6)
        2

        """
        ind = np.where(self.conn == n)[0][0]
        return ind

    def jacobian(self, xyz, dN_Xi):
        """Creates the Jacobian matrix of the mapping between an element

        Arguments
        ---------
            xyz (array of floats): coordinates of element nodes in cartesian
                coordinates
            dN_Xi (array of floats): derivative of shape functions
                dN_Xi = [[dN1_xi, dN2_xi, ...],
                         [dN1_eta, dN2_eta, ...]]

        Returns
        -------
            det_jac (float): determinant of the jacobian matrix
            dN_xi (array of floats): derivative of shape function
                with respect to cartesian system.
                    dN_xi = [[dN1_x, dN2_x, ...],
                             [dN1_y, dN2_y, ...]]
            arch_length (array of floats): arch length for change of variable
                in the line integral

        Note:
            The jacobian matrix is obtained With the chain rule,

                d/dx = (d/dxi)(dxi/dx) + (d/deta)(deta/dx)
                d/dy = (d/dxi)(dxi/dy) + (d/deta)(deta/dy)

            which in matrix form,

                d/dX = J d/dXi

            where X=(x, y) and Xi=(xi, eta) and J is the Jacobian matrix.

                J = [[dxi/dx, deta/dx],
                     [dxi/dy, deta/dy]]

            Inverting the relation above,

                d/dXi = J^{-1} d/dX

            where,

                J^{-1} = [[dx/dxi, dy/dxi],
                          [dx/deta, dy/deta]]
        """
        # jac = [ x1_e1 x2_e1
        #         x1_e2 x2_e2 ]
        jac = dN_Xi @ xyz
        # if (jac[0, 0]*jac[1, 1] - jac[0, 1]*jac[1, 0]) < 0:
        #     print('Negative Jacobiano in element {}'.format(self.eid))

        det_jac = abs((jac[0, 0]*jac[1, 1] -
                       jac[0, 1]*jac[1, 0]))

        # jac_inv = [ e1_x1 e2_x1
        #            e1_x2 e2_x2 ]
        jac_inv = np.linalg.inv(jac)

        # Using Chain rule,
        # N_xi = N_eI * eI_xi (2x4 array)
        dN_xi = np.zeros((2, 4))
        dN_xi[0, :] = (dN_Xi[0, :]*jac_inv[0, 0] +
                       dN_Xi[1, :]*jac_inv[0, 1])

        dN_xi[1, :] = (dN_Xi[0, :]*jac_inv[1, 0] +
                       dN_Xi[1, :]*jac_inv[1, 1])

        # Length of the transofmation arch
        # Jacobian for line integral-2.
        arch_length = np.array([
            (jac[0, 0]**2 + jac[0, 1]**2)**(1/2),
            (jac[1, 0]**2 + jac[1, 1]**2)**(1/2),
            (jac[0, 0]**2 + jac[0, 1]**2)**(1/2),
            (jac[1, 0]**2 + jac[1, 1]**2)**(1/2)
        ])
        return det_jac, dN_xi, arch_length

    def stiffness_matrix(self, t=1):
        """Build the element stiffness matrix

        """
        k = np.zeros((8, 8))

        for w, gp in zip(self.gauss_quad.weights, self.gauss_quad.points):
            N, dN_Xi = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_Xi)
            C = self.c_matrix(N, t)
            B = self.standard_gradient_operator(dN_xi)
            k += w * (B.T @ C @ B) * dJ
        return k * self.thickness

    def standard_gradient_operator(self, dN_xi):
        """Build the standard gradient operator

        Arguments
        ---------
            dN_xi: derivative of shape functions with respect to cartesian
                coordinates

        Returns
        -------
            numpy array shape (3x8)

        Note:
            The order of the matrix follows the node order in element.xyz,
            which follows the order in element.conn. Therefore, if conn is
            [2, 3, 5, 10], the first node, 2, is mapped to node (-1, -1) in
            the isoparametric domain.

            The node tag in conn is not important, but the direction must be
            CCW, so the mapping is consistent and the Jacobian can be computed.

        """
        # standard strain-displacement matrix (discrete gradient operator)
        Bj = {}
        for j in range(self.num_std_nodes):
            Bj[j] = np.array([[dN_xi[0, j], 0],
                             [0, dN_xi[1, j]],
                             [dN_xi[1, j], dN_xi[0, j]]])
        Bstd = np.block([Bj[i] for i in range(self.num_std_nodes)])
        return Bstd

    def mass_matrix(self, t=1):
        """Build element mass matrix

        """
        return None

    def c_matrix(self, N, t=1):
        """Build the element constitutive matrix

        Note:
            Check if E is given as a function, as a list or as a float.

        """
        if callable(self.E):
            x1, x2 = self.mapping(N, self.xyz)
            E = self.E(x1, x2)
        elif type(self.E) is list:
            # interpolate using shape functions
            E = N @ self.E
        else:
            E = self.E

        if type(self.nu) is list:
            nu = N @ self.nu
        else:
            nu = self.nu

        C = np.zeros((3, 3))
        C[0, 0] = 1.0
        C[1, 1] = 1.0
        C[1, 0] = nu
        C[0, 1] = nu
        C[2, 2] = (1.0 - nu)/2.0
        C = (E/(1.0 - nu**2.0))*C

        return C

    def load_body_vector(self, b_force=None, t=1):
        """Build the element vector due body forces b_force

        """

        pb = np.zeros(8)
        if b_force is not None:
            for w, gp in zip(self.gauss_quad.weights, self.gauss_quad.points):
                N, dN_Xi = self.shape_function(xez=gp)
                dJ, dN_xi, _ = self.jacobian(self.xyz, dN_Xi)

                x1, x2 = self.mapping(N, self.xyz)

                pb[0] += w*N[0]*b_force(x1, x2, t)[0]*dJ
                pb[1] += w*N[0]*b_force(x1, x2, t)[1]*dJ
                pb[2] += w*N[1]*b_force(x1, x2, t)[0]*dJ
                pb[3] += w*N[1]*b_force(x1, x2, t)[1]*dJ
                pb[4] += w*N[2]*b_force(x1, x2, t)[0]*dJ
                pb[5] += w*N[2]*b_force(x1, x2, t)[1]*dJ
                pb[6] += w*N[3]*b_force(x1, x2, t)[0]*dJ
                pb[7] += w*N[3]*b_force(x1, x2, t)[1]*dJ

        return pb * self.thickness

    def load_strain_vector(self, t=1):
        """Build the element vector due initial strain

        """
        gauss_points = xez / np.sqrt(3.0)

        pe = np.zeros(8)
        for gp in gauss_points:
            N, dN_Xi = self.shape_function(xez=gp)
            dJ, dN_xi, _ = self.jacobian(self.xyz, dN_Xi)

            C = self.c_matrix(N, t)

            B = np.array([
                [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                 dN_xi[0, 3], 0],
                [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                 dN_xi[1, 3]],
                [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                 dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            pe += (B.T @ C @ self.eps0)*dJ

        return pe * self.thickness

    def load_traction_vector(self, traction_bc=None, t=1):
        """Build element load vector due traction_bction boundary condition

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

        pt = np.zeros(8)

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
                            N, dN_Xi = self.shape_function(xez=gp[ele_side, w])
                            _, _, arch_length = self.jacobian(self.xyz, dN_Xi)

                            dL = arch_length[ele_side]
                            x1, x2 = self.mapping(N, self.xyz)

                            pt[0] += N[0] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[1] += N[0] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[2] += N[1] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[3] += N[1] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[4] += N[2] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[5] += N[2] * traction_bc(x1, x2, t)[key][1] * dL
                            pt[6] += N[3] * traction_bc(x1, x2, t)[key][0] * dL
                            pt[7] += N[3] * traction_bc(x1, x2, t)[key][1] * dL

                    else:
                        # Catch element that is not at boundary
                        continue

        return pt
