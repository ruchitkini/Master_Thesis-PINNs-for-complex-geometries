"""
Date: 13/03/2024            Last change: 13/03/2024

PINN Model for linear elasticity analysis on cantilever beam, E = 1000 Pa, poisson's ratio v = 0.3, with
Body force of 1 N/m3: lmbd = Ev/(1+v)(1-v), mu = E/2(1+v)

"""
import deepxde as dde
import torch
from deepxde import config, optimizers

dde.config.set_default_float('float32')

# Define Parameters
E = 100
nu = 0.3
lmbd = nu * E / ((1 + nu) * (1 - 2*nu))
mu = 0.5 * E / (1 + nu)

geom = dde.geometry.Cuboid([0, 0, 0], [0.2, 0.2, 1])


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.2)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.2)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)


def boundary_front(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 1)


def boundary_back(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 0)


s11_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=3)
s11_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=3)
s22_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=4)
s22_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=4)
s33_front_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_front, component=5)

s12_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=6)
s12_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=6)
s21_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=6)
s21_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=6)

s23_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=7)
s23_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=7)
s32_front_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_front, component=7)

s13_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=8)
s13_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=8)
s31_front_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_front, component=8)


def pde(x, f):
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)
    E_zz = dde.grad.jacobian(f, x, i=2, j=2)
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    E_yz = 0.5 * (dde.grad.jacobian(f, x, i=1, j=2) + dde.grad.jacobian(f, x, i=2, j=1))
    E_zx = 0.5 * (dde.grad.jacobian(f, x, i=0, j=2) + dde.grad.jacobian(f, x, i=2, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + (E_yy + E_zz) * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + (E_xx + E_zz) * lmbd
    S_zz = E_zz * (2 * mu + lmbd) + (E_xx + E_yy) * lmbd
    S_xy = E_xy * 2 * mu
    S_yz = E_yz * 2 * mu
    S_xz = E_zx * 2 * mu

    Sxx_x = dde.grad.jacobian(f, x, i=3, j=0)
    Syy_y = dde.grad.jacobian(f, x, i=4, j=1)
    Szz_z = dde.grad.jacobian(f, x, i=5, j=2)
    Sxy_x = dde.grad.jacobian(f, x, i=6, j=0)
    Sxy_y = dde.grad.jacobian(f, x, i=6, j=1)
    Syz_y = dde.grad.jacobian(f, x, i=7, j=1)
    Syz_z = dde.grad.jacobian(f, x, i=7, j=2)
    Sxz_x = dde.grad.jacobian(f, x, i=8, j=0)
    Sxz_z = dde.grad.jacobian(f, x, i=8, j=2)

    momentum_x = Sxx_x + Sxy_y + Sxz_z
    momentum_y = Sxy_x + Syy_y + Syz_z - 1
    momentum_z = Sxz_x + Syz_y + Szz_z

    stress_x = S_xx - f[:, 3:4]
    stress_y = S_yy - f[:, 4:5]
    stress_z = S_zz - f[:, 5:6]
    stress_xy = S_xy - f[:, 6:7]
    stress_yz = S_yz - f[:, 7:8]
    stress_xz = S_xz - f[:, 8:9]

    return [momentum_x, momentum_y, momentum_z, stress_x, stress_y, stress_z, stress_xy, stress_yz, stress_xz]


data = dde.data.PDE(
    geom,
    pde,
    [s11_left_bc, s11_right_bc, s22_top_bc, s22_bottom_bc, s33_front_bc, s12_left_bc, s12_right_bc, s21_top_bc,
     s21_bottom_bc, s23_top_bc, s23_bottom_bc, s32_front_bc, s13_left_bc, s13_right_bc, s31_front_bc],
    num_domain=20000,
    num_boundary=2000,
    num_test=20000,
)

# layers = [3, [192] * 9, [192] * 9, [192] * 9, [192] * 9, [192] * 9, [192] * 9, 9]
layers = [3, [160] * 9, [160] * 9, [160] * 9, [160] * 9, [160] * 9, [160] * 9, 9]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    ux, uy, uz, sx, sy, sz, sxy, syz, sxz = (f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5], f[:, 5:6],
                                             f[:, 6:7], f[:, 7:8], f[:, 8:9])
    ux_new = z * ux * 1e-3
    uy_new = z * uy * 1e-1
    uz_new = z * uz * 1e-2
    sx_new = sx * 6
    sy_new = sy * 5
    sz_new = sz * 20
    sxy_new = sxy * 0.3
    syz_new = syz * 3
    sxz_new = sxz * 4
    return torch.cat((ux_new, uy_new, uz_new, sx_new, sy_new, sz_new, sxy_new, syz_new, sxz_new), dim=1)


net.apply_output_transform(modify_output)

model = dde.Model(data, net)

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/model", verbose=1, save_better_only=True
)
model.compile("adam", lr=0.001, decay=("inverse time", 50000, 0.91), loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1,
                                              100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])

losshistory, train_state = model.train(display_every=1000, iterations=1800_000, callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
