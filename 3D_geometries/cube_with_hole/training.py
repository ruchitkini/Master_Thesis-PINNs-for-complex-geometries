"""
PINN Model for linear elasticity analysis on a cube with a hole subjected to uniform traction force.
Reference: " A physics-informed neural network technique based on a modified loss function for computational 2D
and 3D solid mechanics "
Link: https://link.springer.com/article/10.1007/s00466-022-02252-0#Sec5
"""
import deepxde as dde
import matplotlib.pyplot as plt
from deepxde import config, optimizers
import numpy as np
import torch
import pandas as pd

E = 10
nu = 0.3
lmbd = nu / ((1 + nu) * (1 - 2*nu))
mu = 0.5 / (1 + nu)

# Create the data for normal vectors
csv_filename1 = 'arc_normals.csv'  # Specify the path to your CSV file
csv_filename2 = 'coords.csv'
csv_filename3 = 'boundary.csv'
csv_filename4 = 'arc.csv'

df1 = pd.read_csv(csv_filename1)  # Read the CSV file into a DataFrame
df2 = pd.read_csv(csv_filename2)
df3 = pd.read_csv(csv_filename3)
df4 = pd.read_csv(csv_filename4)

# Extract specific columns
nv1, nv2, nv3 = df1['xt'], df1['yt'], df1['zt']
xc, yc, zc = df2['x'], df2['y'], df2['z']
xb, yb, zb = df3['xb'], df3['yb'], df3['zb']
xa, ya, za = df4['xa'], df4['ya'], df4['za']

normals = np.zeros((len(nv1), 3))
normals[:, 0], normals[:, 1], normals[:, 2] = nv1, nv2, nv3
coords = np.zeros((len(xc), 3))
coords[:, 0], coords[:, 1], coords[:, 2] = xc, yc, zc
coords_boundary = np.zeros((len(xb), 3))
coords_boundary[:, 0], coords_boundary[:, 1], coords_boundary[:, 2] = xb, yb, zb
coords_boundary_arc = np.zeros((len(xa), 3))
coords_boundary_arc[:, 0], coords_boundary_arc[:, 1], coords_boundary_arc[:, 2] = xa, ya, za

dde.config.set_default_float('float64')

geom = dde.geometry.PointCloud(coords, boundary_points=coords_boundary)
geom_arc = dde.geometry.PointCloud(coords_boundary_arc, boundary_points=coords_boundary_arc)


# Define the boundary functions
def boundary_x0(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)


def boundary_x1(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 1.0)


def boundary_y0(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 0)


def boundary_y1(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[1], 1.0)


def boundary_z0(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 0)


def boundary_z1(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[2], 1.0)


def boundary_arc(x, on_boundary):
    return on_boundary


def traction_arc11(x, f, _):
    ''' 
    The slices implementation here is a naive approach and has not yet been implemented to get the required slices for
    boundary points directly
    '''
    dummy_normal = np.zeros((len(x[:, 0]), 3))
    dummy_normal[7372:8109, 0], dummy_normal[7372:8109, 1], dummy_normal[7372:8109, 2] = (normals[:, 0], normals[:, 1],
                                                                                          normals[:, 2])
    n1, n2, n3 = (torch.as_tensor(dummy_normal[:, 0:1]), torch.as_tensor(dummy_normal[:, 1:2]),
                  torch.as_tensor(dummy_normal[:, 2:3]))
    # n1, n2, n3 = torch.tensor(normals[:, 0:1]), torch.tensor(normals[:, 1:2]), torch.tensor(normals[:, 2:3])
    s11 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=0, j=0)) +
           lmbd * ((dde.grad.jacobian(f, x, i=1, j=1)) + (dde.grad.jacobian(f, x, i=2, j=2))))
    s12 = mu * (dde.grad.jacobian(f, x, i=0, j=1) + dde.grad.jacobian(f, x, i=1, j=0))
    s13 = mu * (dde.grad.jacobian(f, x, i=0, j=2) + dde.grad.jacobian(f, x, i=2, j=0))

    value11 = s11 * n1 + s12 * n2 + s13 * n3

    return value11


def traction_arc22(x, f, _):
    dummy_normal = np.zeros((len(x[:, 0]), 3))
    dummy_normal[8109:8846, 0], dummy_normal[8109:8846, 1], dummy_normal[8109:8846, 2] = (normals[:, 0], normals[:, 1],
                                                                                          normals[:, 2])
    n1, n2, n3 = (torch.as_tensor(dummy_normal[:, 0:1]), torch.as_tensor(dummy_normal[:, 1:2]),
                  torch.as_tensor(dummy_normal[:, 2:3]))
    # n1, n2, n3 = torch.tensor(normals[:, 0:1]), torch.tensor(normals[:, 1:2]), torch.tensor(normals[:, 2:3])
    s22 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=1, j=1)) +
           lmbd * ((dde.grad.jacobian(f, x, i=0, j=0)) + (dde.grad.jacobian(f, x, i=2, j=2))))
    s23 = mu * (dde.grad.jacobian(f, x, i=1, j=2) + dde.grad.jacobian(f, x, i=2, j=1))
    s21 = mu * (dde.grad.jacobian(f, x, i=1, j=0) + dde.grad.jacobian(f, x, i=0, j=1))

    value22 = s21 * n1 + s22 * n2 + s23 * n3

    return value22


def traction_arc33(x, f, _):
    dummy_normal = np.zeros((len(x[:, 0]), 3))
    dummy_normal[8846:9583, 0], dummy_normal[8846:9583, 1], dummy_normal[8846:9583, 2] = (normals[:, 0], normals[:, 1],
                                                                                          normals[:, 2])
    n1, n2, n3 = (torch.as_tensor(dummy_normal[:, 0:1]), torch.as_tensor(dummy_normal[:, 1:2]),
                  torch.as_tensor(dummy_normal[:, 2:3]))
    # n1, n2, n3 = torch.tensor(normals[:, 0:1]), torch.tensor(normals[:, 1:2]), torch.tensor(normals[:, 2:3])
    s33 = ((2 * mu + lmbd) * (dde.grad.jacobian(f, x, i=2, j=2)) +
           lmbd * ((dde.grad.jacobian(f, x, i=1, j=1)) + (dde.grad.jacobian(f, x, i=0, j=0))))
    s32 = mu * (dde.grad.jacobian(f, x, i=2, j=1) + dde.grad.jacobian(f, x, i=1, j=2))
    s31 = mu * (dde.grad.jacobian(f, x, i=2, j=0) + dde.grad.jacobian(f, x, i=0, j=2))

    value33 = s31 * n1 + s32 * n2 + s33 * n3

    return value33


# Define boundary conditions
s11_x1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_x1, component=3)
s22_y1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_y1, component=4)
s33_z1_bc = dde.icbc.DirichletBC(geom, lambda x: 1/E, boundary_z1, component=5)

s12_x0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_x0, component=6)
s12_x1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_x1, component=6)
s21_y0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_y0, component=6)
s21_y1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_y1, component=6)

s23_y0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_y0, component=7)
s23_y1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_y1, component=7)
s32_z0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_z0, component=7)
s32_z1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_z1, component=7)

s13_x0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_x0, component=8)
s13_x1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_x1, component=8)
s31_z0_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_z0, component=8)
s31_z1_bc = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_z1, component=8)

s1_arc_bc = dde.icbc.OperatorBC(geom_arc, traction_arc11, boundary_arc)
s2_arc_bc = dde.icbc.OperatorBC(geom_arc, traction_arc22, boundary_arc)
s3_arc_bc = dde.icbc.OperatorBC(geom_arc, traction_arc33, boundary_arc)


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


data = dde.data.PDE(geom, pde, [s11_x1_bc, s22_y1_bc, s33_z1_bc,
                                s12_x0_bc, s12_x1_bc, s21_y0_bc, s21_y1_bc,
                                s23_y0_bc, s23_y1_bc, s32_z0_bc, s32_z1_bc,
                                s13_x0_bc, s13_x1_bc, s31_z0_bc, s31_z1_bc,
                                s1_arc_bc, s2_arc_bc, s3_arc_bc],
                    num_domain=0,
                    num_boundary=0,
                    num_test=7000,
                    anchors=coords)

layers = [3, [25] * 9, [25] * 9, [25] * 9, [25] * 9, [25] * 9, [25] * 9, [25] * 9, 9]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.PFNN(layers, activation, initializer)


def modify_output(X, f):
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]
    ux, uy, uz, sx, sy, sz, sxy, syz, sxz = (f[:, 0:1], f[:, 1:2], f[:, 2:3], f[:, 3:4], f[:, 4:5], f[:, 5:6],
                                             f[:, 6:7], f[:, 7:8], f[:, 8:9])
    ux_new = x * ux
    uy_new = y * uy
    uz_new = z * uz
    sx_new = sx
    sy_new = sy
    sz_new = sz
    sxy_new = sxy
    syz_new = syz
    sxz_new = sxz
    return torch.cat((ux_new, uy_new, uz_new, sx_new, sy_new, sz_new, sxy_new, syz_new, sxz_new), dim=1)


net.apply_output_transform(modify_output)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10,
                                              10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                              10, 10, 10])

# Save the model during training.
checkpointer = dde.callbacks.ModelCheckpoint(
    "model/model", verbose=1, save_better_only=True
)

model.train(display_every=1000, iterations=200_000, callbacks=[checkpointer])
model.compile("adam", lr=0.0001, loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 10,
                                              10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                              10, 10, 10])
model.train(display_every=1000, iterations=200_000, callbacks=[checkpointer])
dde.optimizers.config.set_LBFGS_options(maxiter=10000)
model.compile("L-BFGS")
losshistory, train_state = model.train(callbacks=[checkpointer])
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
