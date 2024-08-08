from prediction_roller import model, E, data
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os
import pandas as pd
import time
pv.QtBinding = "PyQt5"


u = model.predict(data.train_x_all)
ux = u[:, 0]
plt.scatter(data.train_x_all[:, 0], data.train_x_all[:, 1], c=ux, cmap="jet")
plt.axis('equal')
plt.show()

# ---------- Visualization of the results on the arc for different methods --------- #
# Read the CSV file into a DataFrame
df = pd.read_csv('results/abaqus_roller.csv')  # Read the CSV file into a DataFrame

# Extract specific columns
X, Y, Z = df['X'].values.flatten(), df['Y'].values.flatten(), df['Z'].values.flatten()
U, Ux, Uy = df['U-Magnitude'].values.flatten(), df['U-U1'].values.flatten(), df['U-U2'].values.flatten()
Sxx, Syy, Sxy, S_mises = (df['S-S11'].values.flatten(), df['S-S22'].values.flatten(), df['S-S12'].values.flatten(),
                          df['S-Mises'].values.flatten())
# ---------------------------------------------------------------------------------- #

# ********** Interactive Plot ********** #
# Set the QT_API environment variable
os.environ['QT_API'] = 'pyqt5'

# ######################################################################################### #
# ######################        VISUALIZATION FROM ABAQUS          ######################## #
# ######################################################################################### #
x_pred = np.zeros((len(X), 2))
x_pred[:, 0], x_pred[:, 1] = X, Y

# ---------- prediction - hard constraints ---------- #
start_pred = time.time()
u_pred = model.predict(x_pred)
end_pred = time.time()
pred_time = end_pred - start_pred  # prediction time

ux_pred = (u_pred[:, 0:1]).reshape(-1)
uy_pred = (u_pred[:, 1:2]).reshape(-1)
sxx_pred = (u_pred[:, 2:3] * E).reshape(-1)
syy_pred = (u_pred[:, 3:4] * E).reshape(-1)
sxy_pred = (u_pred[:, 4:5] * E).reshape(-1)

s_mises = (0.5*(sxx_pred - syy_pred)**2 + 3 * sxy_pred**2)**0.5

# absolute error
ux_error = np.abs(ux_pred - Ux)
uy_error = np.abs(uy_pred - Uy)
sxx_error = np.abs(sxx_pred - Sxx)
syy_error = np.abs(syy_pred - Syy)
sxy_error = np.abs(sxy_pred - Sxy)
s_mises_error = np.abs(s_mises - S_mises)

# mean absolute error in %
ux_mae = np.sum(ux_error)/len(x_pred) * 100
uy_mae = np.sum(uy_error)/len(x_pred) * 100
sxx_mae = np.sum(sxx_error)/len(x_pred) * 100
syy_mae = np.sum(syy_error)/len(x_pred) * 100
sxy_mae = np.sum(sxy_error)/len(x_pred) * 100
s_mises_mae = np.sum(s_mises_error)/len(x_pred) * 100

# ----------- PyVista Visualization ---------- #
x_cloud = np.zeros((len(X), 3))
x_cloud[:, 0], x_cloud[:, 1], x_cloud[:, 2] = X, Y, Z

cloud_pred = pv.PolyData(x_cloud)
surface = cloud_pred.delaunay_2d(alpha=0.01)


def result_plot(value, sol_name, save_path1):
    """ This is a function using the pyvista library to visualize the results
    Arguments:
                value - scalar values of the results to plot
                sol_name - string of name of the plot on colorbar
    """

    plotter = pv.Plotter(window_size=[1300, 768])
    plotter.add_mesh(surface, style='surface', show_edges=False, scalars=value, cmap='jet',
                     scalar_bar_args={'title': sol_name, 'vertical': 'True', 'title_font_size': 40,
                                      'label_font_size': 35,
                                      'width': 0.1, 'height': 0.70, 'position_x': 0.75, 'position_y': 0.15})
    plotter.add_axes(interactive=True, line_width=5)
    plotter.add_bounding_box(color='grey')
    # plotter.add_points(cloud, scalars=value, point_size=10, cmap="jet")
    plotter.enable_parallel_projection()
    plotter.camera_position = 'xy'
    plotter.show_grid(grid=None, location='outer', ticks='outside', n_xlabels=3, n_ylabels=3, bold=False,
                      xtitle='', ytitle='', font_size=20)
    # plotter.save_graphic(save_path1)
    plotter.show()


# ---------- Call visualization function (hard constraints) ---------- #
# ux
result_plot(value=ux_pred, sol_name='Ux-PINN', save_path1="results/training_roller/ux_pred.svg")
result_plot(value=Ux, sol_name='Ux-FEM', save_path1="results/training_roller/ux_fem.svg")
result_plot(value=ux_error, sol_name='Ux-Error', save_path1="results/training_roller/ux_error.svg")
# uy
result_plot(value=uy_pred, sol_name='Uy-PINN', save_path1="results/training_roller/uy_pred.svg")
result_plot(value=Uy, sol_name='Uy-FEM', save_path1="results/training_roller/uy_fem.svg")
result_plot(value=uy_error, sol_name='Uy-Error', save_path1="results/training_roller/uy_error.svg")
# sxx
result_plot(value=sxx_pred, sol_name='Sxx-PINN', save_path1="results/training_roller/sxx_pred.svg")
result_plot(value=Sxx, sol_name='Sxx-FEM', save_path1="results/training_roller/sxx_fem.svg")
result_plot(value=sxx_error, sol_name='Sxx-Error', save_path1="results/training_roller/sxx_error.svg")
# syy
result_plot(value=syy_pred, sol_name='Syy-PINN', save_path1="results/training_roller/syy_pred.svg")
result_plot(value=Syy, sol_name='Syy-FEM', save_path1="results/training_roller/syy_fem.svg")
result_plot(value=syy_error, sol_name='Syy-Error', save_path1="results/training_roller/syy_error.svg")
# sxy
result_plot(value=sxy_pred, sol_name='Sxy-PINN', save_path1="results/training_roller/sxy_pred.svg")
result_plot(value=Sxy, sol_name='Sxy-FEM', save_path1="results/training_roller/sxy_fem.svg")
result_plot(value=sxy_error, sol_name='Sxy-Error', save_path1="results/training_roller/sxy_error.svg")

# s_mises
result_plot(value=s_mises, sol_name='S-Mises-PINN', save_path1="results/training_roller/s_mises_pred.svg")
result_plot(value=S_mises, sol_name='S-Mises-FEM', save_path1="results/training_roller/s_mises_fem.svg")
result_plot(value=s_mises_error, sol_name='S-Mises-Error', save_path1="results/training_roller/s_mises_error.svg")
