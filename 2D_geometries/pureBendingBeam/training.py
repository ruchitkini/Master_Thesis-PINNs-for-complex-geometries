"""
PINN Model for linear elasticity analysis of a Pure Bending Beam using displacement boundary conditions,
and force at end of the beam as in the paper " A physics-informed neural network technique based on a modified loss
function for computational 2D and 3D solid mechanics " : https://doi.org/10.1007/s00466-022-02252-0
"""

import deepxde as dde
import numpy as np
import torch

# Define Parameters
E = 1000
nu = 0.3
lmbd = nu * E /((1 + nu) * (1 - 2 * nu))
mu = 0.5 * E /(1+nu)

xmin, xmax = 0.0, 0.5
ymin, ymax = -0.05, 0.05

# Define the geometry
geom = dde.geometry.Rectangle([xmin, ymin], [xmax, ymax])
points_domain = geom.random_points(1000)
points_boundary = geom.random_boundary_points(500)
point1 = np.array([[0, 0]])
point2 = np.append(points_domain, points_boundary, axis=0)
points = np.append(point2, point1, axis=0)


# Define the boundary functions
def boundary(x, on_boundary):
    return on_boundary


def boundary_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.5)


def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0.05)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], -0.05)


def boundary_point(x, on_boundary):
    return on_boundary and (dde.utils.isclose(x[0], 0) and dde.utils.isclose(x[1], 0))


# Define boundary conditions
uy_fixed_bc = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_point, component=1)
sx_right_bc = dde.icbc.DirichletBC(geom, lambda x: 1000 * x[:, 1:2], boundary_right, component=2)
sy_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=3)
sy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=3)
sxy_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=4)
sxy_bottom_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_bottom, component=4)
sxy_right_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right, component=4)
sxy_left_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left, component=4)


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
data = dde.data.PDE(geom, pde, [uy_fixed_bc, sx_right_bc, sy_top_bc, sy_bottom_bc, sxy_top_bc, sxy_bottom_bc, sxy_right_bc, sxy_left_bc],
                    num_domain=0, num_boundary=0, num_test=5000, anchors=points)

# Define the Neural Network
layers = [2, [20] * 5, [20] * 5, [20] * 5, [20] * 5, [20] * 5, 5]
# layers = [2] + [80] * 5 + [5]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y = X[:, 0:1], X[:, 1:2]
    ux, uy, sx, sy, sxy = f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5]
    ux_new = x * ux
    uy_new = uy
    sx_new = sx * 10
    sy_new = sy * 1e-16
    sxy_new = sxy * 1e-16
    return torch.cat((ux_new, uy_new, sx_new, sy_new, sxy_new), dim=1)


net.apply_output_transform(modify_output)

# Define the model, optimiser and learning rate
model = dde.Model(data, net)

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint("model/model", verbose=1, save_better_only=True)

model.compile("adam", lr=0.001, loss_weights=[0.01, 0.01, 0.01, 0.01, 0.01, 1, 1, 1, 1, 1, 1, 1, 1])
model.train(iterations=1000, display_every=1000, callbacks=[checkpointer])
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS")
losshistory, train_state = model.train(callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)