"""
Date: 21/12/2023            Last change: 15/01/2024

PINN Model for linear elasticity analysis on simple cantilever beam, E = 100 Pa, poisson's ratio v = 0.3, with
uniform pressure of 1 N/m2 on right end: lmbd = Ev/(1+v)(1-2v), mu = E/2(1+v)

BC:     (left)          (right)         (bottom)        (top)
        ux = 0                                          sy = -1
        uy = 0

        Modelled as hard constraint using continuous function: ux = x * ux(NN), uy = x * uy(NN)
"""
import torch
import deepxde as dde
import numpy as np
from deepxde import config, optimizers

dde.config.set_default_float('float64')

E = 1000
nu = 0.3
lmbd = nu * E / ((1 + nu) * (1 - 2*nu))
mu = 0.5 * E / (1 + nu)

# Define the geometry
geom = dde.geometry.Rectangle([0, 0], [1, 0.2])


# Define the boundary functions
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.2)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)


def func(x):
    return 0


# Define boundary conditions
s11_right_bc = dde.icbc.DirichletBC(geom, func, on_boundary=boundary_right, component=2)
s22_top_bc = dde.icbc.DirichletBC(geom, lambda x: -1, boundary_top, component=3)
s22_bottom_bc = dde.icbc.DirichletBC(geom, func, on_boundary=boundary_bottom, component=3)
s12_right_bc = dde.icbc.DirichletBC(geom, func, on_boundary=boundary_right, component=4)
s21_top_bc = dde.icbc.DirichletBC(geom, func, on_boundary=boundary_top, component=4)
s21_bottom_bc = dde.icbc.DirichletBC(geom, func, on_boundary=boundary_bottom, component=4)


def pde(x, f):
    ux, uy = f[:, 0:1], f[:, 1:2]
    E_xx = dde.grad.jacobian(f, x, i=0, j=0)
    E_yy = dde.grad.jacobian(f, x, i=1, j=1)
    E_xy = 0.5 * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))

    S_xx = E_xx * (2 * mu + lmbd) + E_yy * lmbd
    S_yy = E_yy * (2 * mu + lmbd) + E_xx * lmbd
    S_xy = E_xy * 2 * mu

    Sxx_x = dde.grad.jacobian(f, x, i=2, j=0)
    Syy_y = dde.grad.jacobian(f, x, i=3, j=1)
    Sxy_x = dde.grad.jacobian(f, x, i=4, j=0)
    Sxy_y = dde.grad.jacobian(f, x, i=4, j=1)

    momentum_x = Sxx_x + Sxy_y
    momentum_y = Sxy_x + Syy_y

    stress_x = S_xx - f[:, 2:3]
    stress_y = S_yy - f[:, 3:4]
    stress_xy = S_xy - f[:, 4:5]

    return [momentum_x, momentum_y, stress_x, stress_y, stress_xy]


data = dde.data.PDE(
    geom,
    pde,
    [s11_right_bc, s22_top_bc, s22_bottom_bc, s12_right_bc, s21_top_bc, s21_bottom_bc],
    num_domain=6000,
    num_boundary=1000,
    num_test=10000,
)

layers = [2, [40] * 5, [40] * 5, [40] * 5, [40] * 5, [40] * 5, 5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y = X[:, 0:1], X[:, 1:2]
    ux, uy, sx, sy, sxy = f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5]
    ux_new = x * ux 
    uy_new = x * uy
    sx_new = sx * 100
    sy_new = sy * 10
    sxy_new = sxy * 10
    return torch.cat((ux_new, uy_new, sx_new, sy_new, sxy_new), dim=1)


net.apply_output_transform(modify_output)

model = dde.Model(data, net)

model.compile("adam", lr=0.0005, loss_weights=[1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10])

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/model", verbose=1, save_better_only=True
)

model.train(display_every=1000, iterations=300_000, callbacks=[checkpointer])

dde.optimizers.config.set_LBFGS_options(maxiter=1200_000)
model.compile("L-BFGS")
losshistory, train_state = model.train(callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
