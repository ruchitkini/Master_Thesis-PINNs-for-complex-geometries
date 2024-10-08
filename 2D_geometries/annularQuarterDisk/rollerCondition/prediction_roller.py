"""
Date: 18/04/2024            Last change: 18/04/2024

"""
import deepxde as dde
import numpy as np
import torch
from deepxde import config, optimizers

dde.config.set_default_float('float64')

# Define Parameters
E = 10
nu = 0.3
lmbd = nu / ((1 + nu) * (1 - 2*nu))
mu = 0.5 / (1 + nu)

# Define the geometry
rect = dde.geometry.Rectangle([0, 0], [1.0, 1.0])
circle1 = dde.geometry.Disk([0, 0], 1)
circle2 = dde.geometry.Disk([0, 0], 0.5)
circle = dde.geometry.CSGDifference(circle1, circle2)
geom = dde.geometry.CSGIntersection(rect, circle)


# Define the boundary functions
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


def boundary_inner_arc(x, on_boundary):
    return on_boundary and (np.isclose(np.linalg.norm(x - [0, 0], axis=-1), 0.5))


def boundary_outer_arc(x, on_boundary):
    return on_boundary and (np.isclose(np.linalg.norm(x - [0, 0], axis=-1), 1))


# Define traction function
def traction_inner11(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s11 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) + lmbd * (dde.grad.jacobian(f, x, i=1, j=1))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value11 = s11 * torch.cos(theta) + s12 * torch.sin(theta) + torch.cos(theta)/E

    return value11


def traction_inner22(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s22 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) + lmbd * (dde.grad.jacobian(f, x, i=0, j=0))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value22 = s12 * torch.cos(theta) + s22 * torch.sin(theta) + torch.sin(theta)/E

    return value22


def traction_outer11(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s11 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) + lmbd * (dde.grad.jacobian(f, x, i=1, j=1))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value11 = s11 * torch.cos(theta) + s12 * torch.sin(theta)

    return value11


def traction_outer22(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s22 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) + lmbd * (dde.grad.jacobian(f, x, i=0, j=0))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value22 = s12 * torch.cos(theta) + s22 * torch.sin(theta)

    return value22


# Define boundary conditions
s1_inner_bc = dde.icbc.OperatorBC(geom, traction_inner11, boundary_inner_arc)
s2_inner_bc = dde.icbc.OperatorBC(geom, traction_inner22, boundary_inner_arc)
s1_outer_bc = dde.icbc.OperatorBC(geom, traction_outer11, boundary_outer_arc)
s2_outer_bc = dde.icbc.OperatorBC(geom, traction_outer22, boundary_outer_arc)

s12_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=4)
s12_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=4)


# Define governing equations
def pde(x, f):
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


# Define data
data = dde.data.PDE(
    geom,
    pde,
    [s1_inner_bc, s2_inner_bc, s1_outer_bc, s2_outer_bc, s12_left_bc, s12_bottom_bc],
    num_domain=1000,
    num_boundary=500,
    num_test=5000)

# Define the Neural Network
layers = [2, [32] * 5, [32] * 5, [32] * 5, [32] * 5, [32] * 5, 5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y = X[:, 0:1], X[:, 1:2]
    ux, uy, sx, sy, sxy = f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5]
    ux_new = x * ux
    uy_new = y * uy
    sx_new = sx
    sy_new = sy
    sxy_new = sxy
    return torch.cat((ux_new, uy_new, sx_new, sy_new, sxy_new), dim=1)


net.apply_output_transform(modify_output)

# Define the model, optimiser and learning rate
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10])
model.compile("L-BFGS")

# Restore the saved model with the smallest training loss
model.restore(f"model_roller/model-40000.pt", verbose=1)
