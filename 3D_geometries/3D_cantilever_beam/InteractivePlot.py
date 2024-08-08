from prediction import model, dde, data, E
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

import pyvista as pv
# from pyvistaqt import BackgroundPlotter
import os
pv.QtBinding = "PyQt5"

# ********** Interactive Plot ********** #


# Set the QT_API environment variable
os.environ['QT_API'] = 'pyqt5'

# plotting
x0 = np.linspace(0, 0.2, 20)
y0 = np.linspace(0, 0.2, 20)
z0 = np.linspace(0, 1, 100)

xs, ys, zs = np.meshgrid(x0, y0, z0)

X = np.zeros((len(x0)*len(y0)*len(z0), 3))
X[:, 0] = xs.reshape(-1)
X[:, 1] = ys.reshape(-1)
X[:, 2] = zs.reshape(-1)

x = X[:, 0]
y = X[:, 1]
z = X[:, 2]

# ######################################################################################### #
# ######################        VISUALIZATION FROM ABAQUS          ######################## #
# ######################################################################################### #

df = pd.read_csv('results/abaqus_fine2.csv')  # Read the CSV file into a DataFrame

# Extract specific columns
X, Y, Z = df['X'].values.flatten(), df['Y'].values.flatten(), df['Z'].values.flatten()
U, Ux, Uy, Uz = df['U-Magnitude'], df['U-U1'], df['U-U2'], df['U-U3']
S_Mises, Sxx, Syy, Szz = df['S-Mises'], df['S-S11'], df['S-S22'], df['S-S33']
Sxy, Syz, Sxz = df['S-S12'], df['S-S23'], df['S-S13']

x_pred = np.zeros((len(X), 3))
x_pred[:, 0], x_pred[:, 1], x_pred[:, 2] = X, Y, Z

# ---------- prediction ---------- #
start_pred = time.time()
u_pred = model.predict(x_pred)
end_pred = time.time()
pred_time = end_pred - start_pred  # prediction time

ux_pred = (u_pred[:, 0:1]).reshape(-1)
uy_pred = (u_pred[:, 1:2]).reshape(-1)
uz_pred = (u_pred[:, 2:3]).reshape(-1)
sxx_pred = (u_pred[:, 3:4]).reshape(-1)
syy_pred = (u_pred[:, 4:5]).reshape(-1)
szz_pred = (u_pred[:, 5:6]).reshape(-1)
sxy_pred = (u_pred[:, 6:7]).reshape(-1)
syz_pred = (u_pred[:, 7:8]).reshape(-1)
sxz_pred = (u_pred[:, 8:9]).reshape(-1)

# von mises stress
s_mises = (0.5 * ((sxx_pred - syy_pred)**2 + (syy_pred - szz_pred)**2 + (szz_pred - sxx_pred)**2) +
           3 * (sxy_pred**2 + syz_pred**2 + sxz_pred**2))**0.5

# absolute error
ux_error = np.abs(ux_pred - Ux)
uy_error = np.abs(uy_pred - Uy)
uz_error = np.abs(uz_pred - Uz)
sxx_error = np.abs(sxx_pred - Sxx)
syy_error = np.abs(syy_pred - Syy)
szz_error = np.abs(szz_pred - Szz)
sxy_error = np.abs(sxy_pred - Sxy)
syz_error = np.abs(syz_pred - Syz)
sxz_error = np.abs(sxz_pred - Sxz)
s_mises_error = np.abs(s_mises - S_Mises)

# mean absolute error in %
ux_mae = np.sum(ux_error)/len(x_pred) * 100
uy_mae = np.sum(uy_error)/len(x_pred) * 100
uz_mae = np.sum(uz_error)/len(x_pred) * 100
sxx_mae = np.sum(sxx_error)/len(x_pred) * 100
syy_mae = np.sum(syy_error)/len(x_pred) * 100
szz_mae = np.sum(szz_error)/len(x_pred) * 100
sxy_mae = np.sum(sxy_error)/len(x_pred) * 100
syz_mae = np.sum(syz_error)/len(x_pred) * 100
sxz_mae = np.sum(sxz_error)/len(x_pred) * 100
s_mises_mae = np.sum(s_mises_error)/len(x_pred) * 100

# ----------- PyVista Visualization ---------- #
points_X = x_pred  # points

cloud = pv.PolyData(points_X)
surface = cloud.delaunay_3d()


def result_plot(value, sol_name, save_path):
    """ This is a function using the pyvista library to visualize the results
    Arguments:
                value - scalar values of the results to plot
                sol_name - string of name of the plot on colorbar
    """

    plotter = pv.Plotter(window_size=[1300, 780])
    plotter.add_mesh(surface, style='surface', show_edges=False, scalars=value, cmap='jet',
                     scalar_bar_args={'title': sol_name, 'vertical': 'True', 'title_font_size': 40,
                                      'label_font_size': 35,
                                      'width': 0.08, 'height': 0.60, 'position_x': 0.685, 'position_y': 0.20})
    plotter.add_axes(interactive=True, line_width=5)
    # plotter.add_bounding_box(color='grey')
    # plotter.add_points(cloud, scalars=value, point_size=10, cmap="jet")
    plotter.enable_parallel_projection()
    plotter.camera_position = 'xy'
    # plotter.camera.roll = 30
    plotter.camera.azimuth = -160
    plotter.camera.elevation = 25
    plotter.show_grid(grid=None, location='outer', ticks='outside', n_xlabels=3, n_ylabels=3, n_zlabels=3, bold=False,
                      xtitle='', ytitle='', ztitle='', font_size=20)
    plotter.save_graphic(save_path)
    # plotter.save_graphic(save_path2)
    plotter.show()


# ---------- Call visualization function (hard constraints) ---------- #
# ux
result_plot(value=ux_pred, sol_name='Ux-PINN', save_path="results/training/ux_pred.svg")
result_plot(value=Ux, sol_name='Ux-FEM', save_path="results/training/ux_fem.svg")
result_plot(value=ux_error, sol_name='Ux-Error', save_path="results/training/ux_error.svg")
# uy
result_plot(value=uy_pred, sol_name='Uy-PINN', save_path="results/training/uy_pred.svg")
result_plot(value=Uy, sol_name='Uy-FEM', save_path="results/training/uy_fem.svg")
result_plot(value=uy_error, sol_name='Uy-Error', save_path="results/training/uy_error.svg")
# uz
result_plot(value=uz_pred, sol_name='Uz-PINN', save_path="results/training/uz_pred.svg")
result_plot(value=Uz, sol_name='Uz-FEM', save_path="results/training/uz_fem.svg")
result_plot(value=uz_error, sol_name='Uz-Error', save_path="results/training/uz_error.svg")
# sxx
result_plot(value=sxx_pred, sol_name='Sxx-PINN', save_path="results/training/sxx_pred.svg")
result_plot(value=Sxx, sol_name='Sxx-FEM', save_path="results/training/sxx_fem.svg")
result_plot(value=sxx_error, sol_name='Sxx-Error', save_path="results/training/sxx_error.svg")
# syy
result_plot(value=syy_pred, sol_name='Syy-PINN', save_path="results/training/syy_pred.svg")
result_plot(value=Syy, sol_name='Syy-FEM', save_path="results/training/syy_fem.svg")
result_plot(value=syy_error, sol_name='Syy-Error', save_path="results/training/syy_error.svg")
# szz
result_plot(value=szz_pred, sol_name='Szz-PINN', save_path="results/training/szz_pred.svg")
result_plot(value=Szz, sol_name='Szz-FEM', save_path="results/training/szz_fem.svg")
result_plot(value=szz_error, sol_name='Szz-Error', save_path="results/training/szz_error.svg")
# sxy
result_plot(value=sxy_pred, sol_name='Sxy-PINN', save_path="results/training/sxy_pred.svg")
result_plot(value=Sxy, sol_name='Sxy-FEM', save_path="results/training/sxy_fem.svg")
result_plot(value=sxy_error, sol_name='Sxy-Error', save_path="results/training/sxy_error.svg")
# syz
result_plot(value=syz_pred, sol_name='Syz-PINN', save_path="results/training/syz_pred.svg")
result_plot(value=Syz, sol_name='Syz-FEM', save_path="results/training/syz_fem.svg")
result_plot(value=syz_error, sol_name='Syz-Error', save_path="results/training/syz_error.svg")
# sxz
result_plot(value=sxz_pred, sol_name='Sxz-PINN', save_path="results/training/sxz_pred.svg")
result_plot(value=Sxz, sol_name='Sxz-FEM', save_path="results/training/sxz_fem.svg")
result_plot(value=sxz_error, sol_name='Sxz-Error', save_path="results/training/sxz_error.svg")
# s_mises
result_plot(value=s_mises, sol_name='S-Mises-PINN', save_path="results/training/s_mises_pred.svg")
result_plot(value=S_Mises, sol_name='S-Mises-FEM', save_path="results/training/s_mises_fem.svg")
result_plot(value=s_mises_error, sol_name='S-Mises-Error', save_path="results/training/s_mises_error.svg")

# cloud = pv.PolyData(data.train_x_all)

plotter = pv.Plotter(window_size=[1300, 780])
plotter.add_points(cloud, scalars=sxy_pred, point_size=2, cmap="jet")
plotter.enable_parallel_projection()
plotter.show()

plotter = pv.Plotter(window_size=[1300, 780])
plotter.add_points(cloud, scalars=Sxy, point_size=2, cmap="jet")
plotter.enable_parallel_projection()
plotter.show()
