import matplotlib.pyplot as plt
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
    1: [15, 2, 12, 1, 1],       # nodes for dirichlet boundary
    2: [15, 2, 13, 2, 2],
    13: [1, 2, 3, 30, 1, 5],
    14: [1, 2, 3, 30, 5, 2],
    3: [1, 2, 7, 2, 2, 6],      # lines for traction
    4: [1, 2, 7, 2, 6, 3],
    7: [1, 2, 5, 4, 4, 8],
    8: [1, 2, 5, 4, 8, 1],
    9: [3, 2, 11, 10, 1, 5, 9, 8],  # quad elements
    10: [3, 2, 11, 10, 5, 2, 6, 9],
    11: [3, 2, 11, 10, 9, 6, 3, 7],
    12: [3, 2, 11, 10, 8, 9, 7, 4]
}
displ = {12: (0, 0), 13: (None, 0)}
traction = {7: (1, 0), 5: (-1, 0)}
mat = skmech.Material(E={11: 1000}, nu={11: 0.3})
model = skmech.Model(msh, material=mat, displacement_bc=displ,
                     traction=traction)
displ = skmech.statics.solver(model)
sig = skmech.postprocess.stress_recovery(model)
fig, ax = plt.subplots()
skmech.plot.geometry(model.nodes, model.elements, ax)
skmech.plot.deformed(model.nodes, model.elements, displ, ax, magf=100)
plt.show()
