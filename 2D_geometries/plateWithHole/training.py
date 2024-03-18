"""
Date: 21/11/2023            Last change: 09/01/2024

PINN Model for linear elasticity analysis of a Perforated strip plate using plane stress condition as in the paper
" A physics-informed neural network technique based on a modified loss function for computational 2D and 3D
solid mechanics " : https://doi.org/10.1007/s00466-022-02252-0

BC:     ux = 0      (left boundary),    
        uy = 0       --> implimented as hard contraints
        sx = 1      (top boundary)
Traction at remaining boundaries need to be explicitly mentioned: s.n = 0

Optimiser: ADAM (5000 iterations) + L-BFGS (25000 iterations)
"""
import deepxde as dde
import numpy as np
import torch
from deepxde import config, optimizers

dde.config.set_default_float('float64')

# Define Parameters
E = 5
nu = 0.3
lmbd = nu * E / ((1 + nu) * (1 - 2*nu))
mu = 0.5 * E / (1 + nu)

# Define the geometry
rect = dde.geometry.Rectangle([0, 0], [1.0, 1.0])
circle = dde.geometry.Disk([0, 0], 0.2)
geom = dde.geometry.CSGDifference(rect, circle)


# Define the boundary functions
def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.0)


def boundary_arc(x, on_boundary):
    return on_boundary and (x[0] <= 0.2 and x[1] <= 0.2)


def traction_arc11(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s11 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) + lmbd * (dde.grad.jacobian(f, x, i=1, j=1))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value11 = s11 * torch.cos(theta) + s12 * torch.sin(theta)

    return value11


def traction_arc22(x, f, _):
    theta = torch.atan2(x[:, 1:2], x[:, 0:1])
    s22 = (2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) + lmbd * (dde.grad.jacobian(f, x, i=0, j=0))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    value22 = s12 * torch.cos(theta) + s22 * torch.sin(theta)

    return value22


# Define boundary conditions
s11_right_bc = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_right, component=2)
s22_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=3)
s12_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=4)
s12_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=4)
s12_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=4)
s12_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=4)

s1_arc_bc = dde.icbc.OperatorBC(geom, traction_arc11, boundary_arc)
s2_arc_bc = dde.icbc.OperatorBC(geom, traction_arc22, boundary_arc)


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
    [s11_right_bc, s22_top_bc, s12_right_bc, s12_left_bc, s12_top_bc, s12_bottom_bc,
     s1_arc_bc, s2_arc_bc],
    num_domain=4000,
    num_boundary=1000,
    num_test=20000)

# Define the Neural Network
layers = [2, [20] * 5, [20] * 5, [20] * 5, [20] * 5, [20] * 5, 5]
# layers = [2] + [60] * 5 + [2]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y = X[:, 0:1], X[:, 1:2]
    ux, uy, sx, sy, sxy = f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5]
    ux_new = x * ux * 0.3
    uy_new = y * uy * 0.127
    sx_new = sx * 3.6
    sy_new = sy * 1.5
    sxy_new = sxy * 1
    return torch.cat((ux_new, uy_new, sx_new, sy_new, sxy_new), dim=1)


net.apply_output_transform(modify_output)

# Define the model, optimiser and learning rate
model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1, 1, 0.1, 0.1, 0.1, 10, 10, 10, 10, 10, 10, 10, 10])

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint("model2/model", verbose=1, save_better_only=True)

# Train the model with Adam and then with BFGS
model.train(iterations=5000, display_every=1000, callbacks=[checkpointer])
dde.optimizers.config.set_LBFGS_options(maxiter=25000)
model.compile("L-BFGS")
losshistory, train_state = model.train(callbacks=[checkpointer])

dde.saveplot(losshistory, train_state, issave=True, isplot=True)

