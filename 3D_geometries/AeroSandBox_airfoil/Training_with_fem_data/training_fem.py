"""
Date: 11/03/2024            Last change: 21/03/2024

PINN Model for linear elasticity analysis on 3D airfoil - NACA0006, E = 1e8 Pa, poisson's ratio v = 0.3, with
traction forces from AeroSandBox: lmbd = Ev/(1+v)(1-2v), mu = E/2(1+v)

### NOTE: USING FEM RESULTS FOR TRAINING
"""

import deepxde as dde
from deepxde import config, optimizers
import torch
import pandas as pd
import numpy as np

# ----- Create the data for coordinates and normal vectors ----- #
csv_filename1 = 'coords.csv'  # Specify the path to your CSV file
csv_filename2 = 'coords_boundary.csv'
csv_filename3 = 'coords_top.csv'
csv_filename4 = 'coords_bottom.csv'
csv_filename5 = 'normals.csv'
csv_filename6 = 'coords_test.csv'

df1 = pd.read_csv(csv_filename1)  # Read the CSV file into a DataFrame
df2 = pd.read_csv(csv_filename2)
df3 = pd.read_csv(csv_filename3)
df4 = pd.read_csv(csv_filename4)
df5 = pd.read_csv(csv_filename5)
df6 = pd.read_csv(csv_filename6)


x_coord, y_coord, z_coord = df1['x'], df1['y'], df1['z']
x_coord_b, y_coord_b, z_coord_b = df2['xb'], df2['yb'], df2['zb']
x_coord_top, y_coord_top, z_coord_top = df3['xb_t'], df3['yb_t'], df3['zb_t']
x_coord_bottom, y_coord_bottom, z_coord_bottom = df4['xb_b'], df4['yb_b'], df4['zb_b']
n1_t, n2_t, n3_t = df5['xt'], df5['yt'], df5['zt']
n1_b, n2_b, n3_b = df5['xb'], df5['yb'], df5['zb']
x_test, y_test, z_test = df6['tx'], df6['ty'], df6['tz']

coords = np.zeros((len(x_coord), 3))
coords[:, 0], coords[:, 1], coords[:, 2] = x_coord, y_coord, z_coord
coords_boundary = np.zeros((len(x_coord_b), 3))
coords_boundary[:, 0], coords_boundary[:, 1], coords_boundary[:, 2] = x_coord_b, y_coord_b, z_coord_b
coords_boundary_top = np.zeros((len(x_coord_top), 3))
coords_boundary_top[:, 0], coords_boundary_top[:, 1], coords_boundary_top[:, 2] = x_coord_top, y_coord_top, z_coord_top
coords_boundary_bottom = np.zeros((len(x_coord_bottom), 3))
coords_boundary_bottom[:, 0], coords_boundary_bottom[:, 1], coords_boundary_bottom[:, 2] = (x_coord_bottom,
                                                                                            y_coord_bottom,
                                                                                            z_coord_bottom)
coords_test = np.zeros((len(x_test), 3))
coords_test[:, 0], coords_test[:, 1], coords_test[:, 2] = x_test, y_test, z_test
nv_top = np.zeros((len(n1_t), 3))
nv_top[:, 0], nv_top[:, 1], nv_top[:, 2] = n1_t, n2_t, n3_t
nv_bottom = np.zeros((len(n1_b), 3))
nv_bottom[:, 0], nv_bottom[:, 1], nv_bottom[:, 2] = n1_b, n2_b, n3_b

dde.config.set_default_float('float64')

# ----- Create random domain points and boundary points ---- #
id_anchor_in = np.random.choice(coords.shape[0], size=10000, replace=False)
anchor_in = coords[id_anchor_in]

id_anchor_bound = np.random.choice(coords_boundary.shape[0], size=2000, replace=False)
anchor_bound = coords_boundary[id_anchor_bound]

anchor = np.vstack((anchor_bound, anchor_in))


# ----- Parameters from FEM Software for training ----- #

def generate_data(num):
    data = pd.read_csv("abaqus_result.csv")

    x_fem, y_fem, z_fem = data['X'].values.flatten(), data['Y'].values.flatten(), data['Z'].values.flatten()
    Ux_fem, Uy_fem, Uz_fem = (data['U-U1'].values.flatten()[:, None], data['U-U2'].values.flatten()[:, None],
                              data['U-U3'].values.flatten()[:, None])
    Sxx_fem, Syy_fem, Szz_fem = (data['S-S11'].values.flatten()[:, None], data['S-S22'].values.flatten()[:, None],
                                 data['S-S33'].values.flatten()[:, None])
    Sxy_fem, Syz_fem, Sxz_fem = (data['S-S12'].values.flatten()[:, None], data['S-S23'].values.flatten()[:, None],
                                 data['S-S13'].values.flatten()[:, None])

    X_fem = np.hstack((x_fem.flatten()[:, None], y_fem.flatten()[:, None], z_fem.flatten()[:, None]))

    num1 = int(num/10)
    region1 = X_fem[:, 2] == 0  # on fixed boundary
    idx1 = np.random.choice(np.where(region1)[0], num1, replace=False)  # sample points

    num2 = int(num/10)
    region2 = X_fem[:, 2] == 1  # on free boundary
    idx2 = np.random.choice(np.where(region2)[0], num2, replace=False)  # sample points

    num3 = int(4*num/10)
    region3 = X_fem[:, 2] <= 0.1  # on near fixed boundary
    idx3 = np.random.choice(np.where(region3)[0], num3, replace=False)  # sample points

    num4 = int(4*num/10)
    region4 = X_fem[:, 2] >= 0.1  # in rest of the domain
    idx4 = np.random.choice(np.where(region4)[0], num4, replace=False)  # sample points

    print("sampled points = ", num1+num2+num3+num4)

    x_star = np.vstack((X_fem[idx1], X_fem[idx2], X_fem[idx3], X_fem[idx4]))
    ux_star = np.vstack((Ux_fem[idx1], Ux_fem[idx2], Ux_fem[idx3], Ux_fem[idx4]))
    uy_star = np.vstack((Uy_fem[idx1], Uy_fem[idx2], Uy_fem[idx3], Uy_fem[idx4]))
    uz_star = np.vstack((Uz_fem[idx1], Uz_fem[idx2], Uz_fem[idx3], Uz_fem[idx4]))
    sxx_star = np.vstack((Sxx_fem[idx1], Sxx_fem[idx2], Sxx_fem[idx3], Sxx_fem[idx4]))
    syy_star = np.vstack((Syy_fem[idx1], Syy_fem[idx2], Syy_fem[idx3], Syy_fem[idx4]))
    szz_star = np.vstack((Szz_fem[idx1], Szz_fem[idx2], Szz_fem[idx3], Szz_fem[idx4]))
    sxy_star = np.vstack((Sxy_fem[idx1], Sxy_fem[idx2], Sxy_fem[idx3], Sxy_fem[idx4]))
    syz_star = np.vstack((Syz_fem[idx1], Syz_fem[idx2], Syz_fem[idx3], Syz_fem[idx4]))
    sxz_star = np.vstack((Sxz_fem[idx1], Sxz_fem[idx2], Sxz_fem[idx3], Sxz_fem[idx4]))

    return x_star, ux_star, uy_star, uz_star, sxx_star, syy_star, szz_star, sxy_star, syz_star, sxz_star


observe_xy_t, ux_t, uy_t, uz_t, sxx_t, syy_t, szz_t, sxy_t, syz_t, sxz_t = generate_data(5000)

# ----- Define Parameters ----- #
E = 1e8
nu = 0.3
lmbd = nu * E / ((1 + nu) * (1 - 2*nu))
mu = 0.5 * E / (1 + nu)

# ----- Create geometries ---- #
geom = dde.geometry.PointCloud(coords, boundary_points=coords_boundary)
geom_test = dde.geometry.PointCloud(coords_test)
geom_top = dde.geometry.PointCloud(coords_boundary_top, boundary_points=coords_boundary_top)
geom_bottom = dde.geometry.PointCloud(coords_boundary_bottom, boundary_points=coords_boundary_bottom)


# ----- Define boundaries ----- #
def boundary_fixed(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 0)


def boundary_free(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 1)


def boundary_top(x, on_boundary):
    return on_boundary


def boundary_bottom(x, on_boundary):
    return on_boundary


def boundary_end(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0.5)


# ----- Define boundary conditions -----#
observe_ux = dde.PointSetBC(observe_xy_t, ux_t, component=0)
observe_uy = dde.PointSetBC(observe_xy_t, uy_t, component=1)
observe_uz = dde.PointSetBC(observe_xy_t, uz_t, component=2)

observe_sxx = dde.PointSetBC(observe_xy_t, sxx_t, component=3)
observe_syy = dde.PointSetBC(observe_xy_t, syy_t, component=4)
observe_szz = dde.PointSetBC(observe_xy_t, szz_t, component=5)

observe_sxy = dde.PointSetBC(observe_xy_t, sxy_t, component=6)
observe_syz = dde.PointSetBC(observe_xy_t, syz_t, component=7)
observe_sxz = dde.PointSetBC(observe_xy_t, sxz_t, component=8)


# ----- Define PDEs ------ #
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


data = dde.data.PDE(geom_test, pde, [observe_ux, observe_uy, observe_uz, observe_sxx, observe_syy, observe_szz,
                                     observe_sxy, observe_syz, observe_sxz],
                    num_domain=0, num_boundary=0, num_test=30000, anchors=anchor)

layers = [3, [60] * 9, [60] * 9, [60] * 9, [60] * 9, [60] * 9, [60] * 9, 9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    ux, uy, uz, sx, sy, sz, sxy, syz, sxz = (f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5], f[:, 5:6],
                                             f[:, 6:7], f[:, 7:8], f[:, 8:9])
    ux_new = z * ux * 1.14e-5
    uy_new = z * uy * 2.71e-3
    uz_new = z * uz * 5.24e-5
    sx_new = sx * 5.3e3
    sy_new = sy * 1.77e3
    sz_new = sz * 1.66e4
    sxy_new = sxy * 490
    syz_new = syz * 1.43e3
    sxz_new = sxz * 1.1e3

    return torch.cat((ux_new, uy_new, uz_new, sx_new, sy_new, sz_new, sxy_new, syz_new, sxz_new), dim=1)


model = dde.Model(data, net)

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint("model_fem/model", verbose=1, save_better_only=True)

model.compile("adam", lr=0.001, loss_weights=[1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15,
                                              10, 10, 10, 1, 1, 1, 1, 1, 1])
model.train(display_every=1000, iterations=1000_000, callbacks=[checkpointer])
model.compile("adam", lr=0.001, loss_weights=[1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
                                              10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
model.train(display_every=1000, iterations=600_000, callbacks=[checkpointer])
model.compile("adam", lr=0.0001, loss_weights=[1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10,
                                              10, 10, 10, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
model.train(display_every=1000, iterations=400_000, callbacks=[checkpointer])

# dde.optimizers.config.set_LBFGS_options(maxiter=5000)
model.compile("L-BFGS")
losshistory, train_state = model.train(callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
