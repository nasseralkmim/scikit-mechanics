import numpy as np
import pytest
import skmech


class Mesh():
    pass


msh = Mesh()
msh.nodes = {
    1: [0, 0, 0],
    2: [1, 0, 0],
    3: [1, 1, 0],
    4: [0, 1, 0],
    5: [.5, 0, 0],
    6: [1, .5, 0],
    7: [.5, 1, 0],
    8: [0, .5, 0],
    9: [.4, .6]
}
msh.elements = {
    1: [15, 2, 12, 1, 1],
    2: [15, 2, 13, 2, 2],
    3: [1, 2, 7, 2, 2, 6],
    4: [1, 2, 7, 2, 6, 3],
    7: [1, 2, 5, 4, 4, 8],
    8: [1, 2, 5, 4, 8, 1],
    9: [3, 2, 11, 10, 1, 5, 9, 8],
    10: [3, 2, 11, 10, 5, 2, 6, 9],
    11: [3, 2, 11, 10, 9, 6, 3, 7],
    12: [3, 2, 11, 10, 8, 9, 7, 4]
}
material = skmech.Material(E={11: 10000}, nu={11: 0.3})
traction = {5: (-1, 0), 7: (1, 0)}
displacement_bc = {12: (0, 0), 13: (None, 0)}
model = skmech.Model(
    msh,
    material=material,
    traction=traction,
    displacement_bc=displacement_bc,
    num_quad_points=2)


def test_tractionbc():
    Pt = skmech.neumann(model)
    assert list(Pt) == [
        -0.25, 0., 0.25, 0., 0.25, 0., -0.25, 0., 0., 0., 0.5, 0., 0., 0.,
        -0.5, 0., 0., 0.
    ]


def test_stiffness():
    K = 0
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        k = ele.stiffness_matrix()
        K += k
    assert np.round(np.linalg.norm(K), 2) == 52268.5


def test_gradient_operator():
    class Mesh():
        pass

    msh = Mesh()
    msh.nodes = {1: [0, 0, 0], 2: [6, 0, 0], 3: [8, 6, 0], 4: [2, 6, 0]}
    msh.elements = {1: [3, 2, 0, 0, 1, 2, 3, 4]}
    material = skmech.material.Material(E={0: 10000}, nu={0: 0.2})
    model = skmech.model.Model(msh, material, num_quad_points=2)

    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        N, dN_ei, dJ, dN_xi, B = {}, {}, {}, {}, {}
        k = 0
        for i, (w, gp) in enumerate(zip(ele.gauss.weights, ele.gauss.points)):
            N[i], dN_ei[i] = ele.shape_function(gp)
            dJ[i], dN_xi[i], _ = ele.jacobian(ele.xyz, dN_ei[i])
            B[i] = ele.gradient_operator(dN_xi[i])
            C = ele.c_matrix(N[i], ele.xyz)
            k += w * dJ[i] * (B[i].T @ C @ B[i])

    u = np.array([0, 0, 0, 0, 1, 0, 0, 0])
    assert pytest.approx(list(k @ u), 4) == [
        -1813.27, -1215.28, 424.38, 173.61, 4320.99, 868.06, -2932.1, 173.61
    ]
    assert pytest.approx(list(N[0]), 4) == [
        0.62200847, 0.16666667, 0.0446582, 0.16666667
    ]
    assert pytest.approx(list(dN_ei[0][0, :]), 4) == [
        -0.3943, 0.3943, 0.1057, -0.1057
    ]
    assert dJ[0] == 9.0
    assert pytest.approx(list(dN_xi[0][0, :]), 4) == [
        -0.1314, 0.1314, 0.0352, -0.0352
    ]


def test_neumann():
    class Mesh():
        pass

    msh = Mesh()
    msh.nodes = {1: [0, 0, 0], 2: [14, 0, 0], 3: [6, 6, 0], 4: [0, 6, 0]}
    msh.elements = {1: [1, 2, 5, 4, 2, 3], 2: [3, 2, 0, 0, 1, 2, 3, 4]}
    mat = skmech.Material(E={0: 10000}, nu={0: 0.2})
    trac = {5: (3 / 5, 4 / 5)}
    model = skmech.Model(mesh=msh, material=mat, traction=trac,
                         num_quad_points=2)
    Pt = skmech.neumann(model)
    assert list(Pt) == [0., 0., 3., 4., 3., 4., 0., 0.]


def test_dirichlet():
    K = np.ones((model.num_dof, model.num_dof))
    F = np.ones(model.num_dof)
    Km, Fm = skmech.dirichlet(K, F, model)
    assert Km[0, 0] == 1
    assert Km[1, 1] == 1
    assert Fm[0] == 0
    assert Fm[1] == 0
