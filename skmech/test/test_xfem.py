import pytest
import skmech


class Mesh():
    pass


msh = Mesh()
msh.nodes = {
    1: [0, 0, 0],
    2: [.2, 0, 0],
    3: [.2, .2, 0],
    4: [0, 0.2, 0],
    5: [.4, 0, 0],
    6: [.6, 0, 0],
    7: [.6, .2, 0],
    8: [.4, .2, 0],
}
msh.elements = {
    1: [15, 2, 12, 1, 1],
    2: [15, 2, 13, 4, 4],
    13: [1, 2, 3, 30, 6, 7],
    3: [1, 2, 7, 2, 3, 7],
    4: [1, 2, 7, 2, 7, 4],
    9: [3, 2, 11, 10, 1, 2, 3, 4],
    10: [3, 2, 11, 10, 2, 5, 8, 3],
    11: [3, 2, 11, 10, 5, 6, 7, 8],
}
func = lambda x, y: x - 0.3
zls = skmech.xfem.ZeroLevelSet(func, [0, .6], [0, .2], num_div=100)
model = skmech.Model(msh, zerolevelset=zls)



def test_xfem_phi():
    func = lambda x, y: x - 0.3
    zls = skmech.xfem.ZeroLevelSet(func, [0, .6], [0, .2], num_div=100)
    xfem = skmech.xfem.Xfem(model.nodes, model.elements, zls, None)
    assert pytest.approx(list(xfem.zls[0].phi), 3) == [
        -0.5, -0.16666667, -0.16666667, -0.5, 0.16666667, 0.5, 0.5, 0.16666667
    ]


def test_xfem_enr_elements():
    func = lambda x, y: x - 0.1
    zls = skmech.xfem.ZeroLevelSet(func, [0, .6], [0, .2], num_div=100)
    model = skmech.Model(msh, zerolevelset=zls)
    assert model.xfem.enr_elements == [9, 10]
