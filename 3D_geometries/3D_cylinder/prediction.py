"""
PINN Model for linear elasticity analysis on a cylinder with subjected to uniform traction force.

"""
import deepxde as dde
import matplotlib.pyplot as plt
from deepxde import config, optimizers
import numpy as np
import torch
import pandas as pd

dde.config.set_default_float('float32')

E = 10
nu = 0.3
lmbd = nu * E / ((1 + nu) * (1 - 2*nu))
mu = 0.5 * E / (1 + nu)

# Create the data for normal vectors
df2 = pd.read_csv('coords_rand.csv')
df3 = pd.read_csv('boundary_rand.csv')

# Extract specific columns
xc, yc, zc = df2['x'], df2['y'], df2['z']
xb, yb, zb = df3['xb'], df3['yb'], df3['zb']

coords = np.zeros((len(xc), 3))
coords[:, 0], coords[:, 1], coords[:, 2] = xc, yc, zc
coords_boundary = np.zeros((len(xb), 3))
coords_boundary[:, 0], coords_boundary[:, 1], coords_boundary[:, 2] = xb, yb, zb

geom = dde.geometry.PointCloud(coords, boundary_points=coords_boundary)
print()
# anchor = np.vstack((coords_boundary, coords))


# Define the boundary functions
def boundary_top(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1)


def boundary_bottom(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)


def boundary_curved(x, on_boundary):
    return on_boundary and (np.isclose(np.linalg.norm(x - [0, x[1], 0], axis=-1), 0.5))


def traction_curve11(x, f, _):
    theta = torch.atan2(x[:, 2:3], x[:, 0:1])
    n1 = torch.where(x[:, 0:1] >= 0, torch.cos(theta), -torch.cos(theta))
    n2 = torch.zeros((len(n1), 1))
    n3 = torch.where(x[:, 0:1] >= 0, torch.sin(theta), -torch.sin(theta))

    s11 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) +
           lmbd * ((dde.grad.jacobian(f, x, i=1, j=1)) + (dde.grad.jacobian(f, x, i=2, j=2))))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    s13 = mu * (dde.grad.jacobian(f, x, i=0, j=2) + dde.grad.jacobian(f, x, i=2, j=0))

    value11 = s11 * n1 + s12 * n2 + s13 * n3

    return value11


def traction_curve22(x, f, _):
    theta = torch.atan2(x[:, 2:3], x[:, 0:1])
    n1 = torch.where(x[:, 0:1] >= 0, torch.cos(theta), -torch.cos(theta))
    n2 = torch.zeros((len(n1), 1))
    n3 = torch.where(x[:, 0:1] >= 0, torch.sin(theta), -torch.sin(theta))

    s22 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) +
           lmbd * ((dde.grad.jacobian(f, x, i=0, j=0)) + (dde.grad.jacobian(f, x, i=2, j=2))))
    s23 = mu * (dde.grad.jacobian(f, x, i=1, j=2) + dde.grad.jacobian(f, x, i=2, j=1))
    s21 = mu * (dde.grad.jacobian(f, x, i=1, j=0) + dde.grad.jacobian(f, x, i=0, j=1))

    value22 = s21 * n1 + s22 * n2 + s23 * n3

    return value22


def traction_curve33(x, f, _):
    theta = torch.atan2(x[:, 2:3], x[:, 0:1])
    n1 = torch.where(x[:, 0:1] >= 0, torch.cos(theta), -torch.cos(theta))
    n2 = torch.zeros((len(n1), 1))
    n3 = torch.where(x[:, 0:1] >= 0, torch.sin(theta), -torch.sin(theta))

    s33 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=2, j=2)) +
           lmbd * ((dde.grad.jacobian(f, x, i=1, j=1)) + (dde.grad.jacobian(f, x, i=0, j=0))))
    s32 = mu * (dde.grad.jacobian(f, x, i=2, j=1) + dde.grad.jacobian(f, x, i=1, j=2))
    s31 = mu * (dde.grad.jacobian(f, x, i=2, j=0) + dde.grad.jacobian(f, x, i=0, j=2))

    value33 = s31 * n1 + s32 * n2 + s33 * n3

    return value33


# Define boundary conditions
s22_top_bc = dde.icbc.DirichletBC(geom, lambda x: -1, boundary_top, component=4)
s21_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=6)
s23_top_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_top, component=7)

s1_curve_bc = dde.icbc.OperatorBC(geom, traction_curve11, boundary_curved)
s2_curve_bc = dde.icbc.OperatorBC(geom, traction_curve22, boundary_curved)
s3_curve_bc = dde.icbc.OperatorBC(geom, traction_curve33, boundary_curved)


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
    momentum_y = Sxy_x + Syy_y + Syz_z
    momentum_z = Sxz_x + Syz_y + Szz_z

    stress_x = S_xx - f[:, 3:4]
    stress_y = S_yy - f[:, 4:5]
    stress_z = S_zz - f[:, 5:6]
    stress_xy = S_xy - f[:, 6:7]
    stress_yz = S_yz - f[:, 7:8]
    stress_xz = S_xz - f[:, 8:9]

    return [momentum_x, momentum_y, momentum_z, stress_x, stress_y, stress_z, stress_xy, stress_yz, stress_xz]


data = dde.data.PDE(geom, pde, [],
                    num_domain=25000,
                    num_boundary=2000,
                    num_test=20000)

layers = [3, [160] * 9, [160] * 9, [160] * 9, [160] * 9, [160] * 9, [160] * 9, 9]
activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    ux, uy, uz, sx, sy, sz, sxy, syz, sxz = (f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5], f[:, 5:6],
                                             f[:, 6:7], f[:, 7:8], f[:, 8:9])
    ux_new = y * ux
    uy_new = y * uy
    uz_new = y * uz
    sx_new = sx
    sy_new = sy
    sz_new = sz
    sxy_new = sxy
    syz_new = syz
    sxz_new = sxz
    return torch.cat((ux_new, uy_new, uz_new, sx_new, sy_new, sz_new, sxy_new, syz_new, sxz_new), dim=1)


net.apply_output_transform(modify_output)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, decay=("inverse time", 20000, 0.912))
model.compile("L-BFGS")

# Restore the saved model with the smallest training loss
model.restore(f"model_random/model-2138000.pt", verbose=1)