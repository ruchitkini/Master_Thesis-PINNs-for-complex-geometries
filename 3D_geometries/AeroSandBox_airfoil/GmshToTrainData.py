import numpy as np
import gmsh
# import sys
import matplotlib.pyplot as plt

import pyvista as pv
from pyvista import examples
# from pyvistaqt import BackgroundPlotter
import os
pv.QtBinding = "PyQt5"
import pandas as pd

gmsh.initialize()
gmsh.open("NACA0006_airfoil_train4.msh")

nodeTags, coords, parametricCoord = gmsh.model.mesh.getNodes(-1, -1)
coords = coords.reshape(-1, 3)
print(coords)

# get boundary coordinates
boundary_node_tags_top, boundary_node_coords_top = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 1)
coords_boundary_top = boundary_node_coords_top.reshape(-1, 3)

boundary_node_tags_bottom, boundary_node_coords_bottom = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 2)
coords_boundary_bottom = boundary_node_coords_bottom.reshape(-1, 3)

# boundary_node_tags_front, boundary_node_coords_front = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 33)
# coords_boundary_front = boundary_node_coords_front.reshape(-1, 3)

boundary_node_tags_end, boundary_node_coords_end = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 5)
coords_boundary_end = boundary_node_coords_end.reshape(-1, 3)

boundary_node_tags_fixedside, boundary_node_coords_fixedside = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 3)
coords_boundary_fixedside = boundary_node_coords_fixedside.reshape(-1, 3)

boundary_node_tags_freeside, boundary_node_coords_freeside = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 4)
coords_boundary_freeside = boundary_node_coords_freeside.reshape(-1, 3)
# print(coords_boundary)

coords_boundary1 = np.vstack((coords_boundary_top, coords_boundary_bottom))
# coords_boundary2 = np.vstack((coords_boundary1, coords_boundary_front))
coords_boundary3 = np.vstack((coords_boundary1, coords_boundary_end))
coords_boundary4 = np.vstack((coords_boundary3, coords_boundary_fixedside))
coords_boundary_all = np.vstack((coords_boundary4, coords_boundary_freeside))

coords_boundary = np.unique(coords_boundary_all, axis=0)
print(f"the length original list is {len(coords_boundary_all)} and the new list without duplicates is "
      f"{len(coords_boundary)}")


gmsh.finalize()


print('coords =', len(coords))
print('coords_boundary_top=', len(coords_boundary_top))
print('coords_boundary_bottom=', len(coords_boundary_bottom))
print('coords_boundary_end=', len(coords_boundary_end))
print('coords_boundary_fixedside=', len(coords_boundary_fixedside))
print('coords_boundary_freeside=', len(coords_boundary_freeside))
print('coords_boundary_total=', len(coords_boundary))


gmsh.initialize()
gmsh.open("NACA0006_airfoil_test4.msh")

nodeTags_test, coords_test, parametricCoord_test = gmsh.model.mesh.getNodes(-1, -1)
coords_test = coords_test.reshape(-1, 3)

boundary_node_tags_top_test, boundary_node_coords_top_test = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 1)
coords_boundary_top_test = boundary_node_coords_top_test.reshape(-1, 3)

boundary_node_tags_bottom_test, boundary_node_coords_bottom_test = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 2)
coords_boundary_bottom_test = boundary_node_coords_bottom_test.reshape(-1, 3)

boundary_node_tags_end_test, boundary_node_coords_end_test = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 5)
coords_boundary_end_test = boundary_node_coords_end_test.reshape(-1, 3)

boundary_node_tags_fixedside_test, boundary_node_coords_fixedside_test = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 3)
coords_boundary_fixedside_test = boundary_node_coords_fixedside_test.reshape(-1, 3)

boundary_node_tags_freeside_test, boundary_node_coords_freeside_test = gmsh.model.mesh.getNodesForPhysicalGroup(-1, 4)
coords_boundary_freeside_test = boundary_node_coords_freeside_test.reshape(-1, 3)

coords_boundary_all_test = np.vstack((coords_boundary_top_test, coords_boundary_bottom_test, coords_boundary_end_test,
                                      coords_boundary_fixedside_test, coords_boundary_freeside_test))
coords_boundary_test = np.unique(coords_boundary_all_test, axis=0)
print(f"the length original list is {len(coords_boundary_all_test)} and the new list without duplicates is "
      f"{len(coords_boundary_test)}")


# Reshape arrays to 2D
coords_test_2d = coords_test.reshape(-1, coords_test.shape[-1])
coords_boundary_test_2d = coords_boundary_test.reshape(-1, coords_boundary_test.shape[-1])
# Convert arrays to sets of tuples
coords_test_set = {tuple(row) for row in coords_test_2d}
coords_boundary_test_set = {tuple(row) for row in coords_boundary_test_2d}
# Perform set difference
result_set_test = coords_test_set - coords_boundary_test_set
# Convert result set back to NumPy array
coords_test_new = np.array(list(result_set_test))

idx = np.random.choice(coords_test_new.shape[0], 50000, replace=False)  # sample points
coords_new = coords_test_new[idx]

coords_domain = coords_new
print("coords_rand: ", len(coords_domain))

gmsh.finalize()

# ********** Interactive Plot ********** #
# Set the QT_API environment variable
os.environ['QT_API'] = 'pyqt5'

x = coords[:, 0]
y = coords[:, 1]
z = coords[:, 2]

x_b = coords_boundary[:, 0]
y_b = coords_boundary[:, 1]
z_b = coords_boundary[:, 2]

x_b_top = coords_boundary_top[:, 0]
y_b_top = coords_boundary_top[:, 1]
z_b_top = coords_boundary_top[:, 2]

x_b_bottom = coords_boundary_bottom[:, 0]
y_b_bottom = coords_boundary_bottom[:, 1]
z_b_bottom = coords_boundary_bottom[:, 2]

x_b_end = coords_boundary_end[:, 0]
y_b_end = coords_boundary_end[:, 1]
z_b_end = coords_boundary_end[:, 2]

x_b_fixedside = coords_boundary_fixedside[:, 0]
y_b_fixedside = coords_boundary_fixedside[:, 1]
z_b_fixedside = coords_boundary_fixedside[:, 2]

x_b_freeside = coords_boundary_freeside[:, 0]
y_b_freeside = coords_boundary_freeside[:, 1]
z_b_freeside = coords_boundary_freeside[:, 2]

# Create a PyVista point cloud
grid = pv.UnstructuredGrid('NACA0006_airfoil_train4.msh')
# grid.plot(show_edges=True)

cloud = pv.PolyData(grid.points)
cloud_test = pv.PolyData(coords_test)

cloud_b = pv.PolyData(np.column_stack((x_b, y_b, z_b)))
cloud_b_top = pv.PolyData(np.column_stack((x_b_top, y_b_top, z_b_top)))
cloud_b_bottom = pv.PolyData(np.column_stack((x_b_bottom, y_b_bottom, z_b_bottom)))
cloud_b_end = pv.PolyData(np.column_stack((x_b_end, y_b_end, z_b_end)))
cloud_b_fixedside = pv.PolyData(np.column_stack((x_b_fixedside, y_b_fixedside, z_b_fixedside)))
cloud_b_freeside = pv.PolyData(np.column_stack((x_b_freeside, y_b_freeside, z_b_freeside)))
cloud_domain = pv.PolyData(coords_domain)
# Perform a Delaunay triangulation
top_bottom = np.row_stack((np.column_stack((x_b_top, y_b_top, z_b_top)), np.column_stack((x_b_bottom, y_b_bottom, z_b_bottom))))
surface = (pv.PolyData(top_bottom)).delaunay_3d()
surface_top = cloud_b_top.delaunay_2d()
surface_bottom = cloud_b_bottom.delaunay_2d()

# Plot the colored volume shape
plotter = pv.Plotter()
# plotter = BackgroundPlotter()   # interactive plot
# To view the 3D figure
#plotter.add_mesh(grid, style='surface', show_edges=False, metallic=1, color='#87CEEB')  # #C0C0C0
plotter.add_points(cloud_test, color="black", point_size=3)
# plotter.add_axes(interactive=True, line_width=5)
# plotter.add_bounding_box(color='black')
# plotter.show_grid()
#cubemap = examples.download_sky_box_cube_map()
#plotter.set_background('#87CEEB', top='white')
#plotter.set_environment_texture(cubemap)
#plotter.render()
#plotter.enable_shadows()
#plotter.show()


# **** points from aero-tool **** #
data = pd.read_csv("traction_data.csv")

x_traction, y_traction, z_traction = (data['x_traction'].values.flatten(),
                                      data['y_traction'].values.flatten(),
                                      data['z_traction'].values.flatten())
pressure = data['pressure'].values.flatten()[:, None]

X_traction = np.hstack((x_traction.flatten()[:, None], y_traction.flatten()[:, None], z_traction.flatten()[:, None]))

id_bottom = np.random.choice(X_traction.shape[0], 1000, replace=False)  # top boundary
bottom_bc_points = X_traction[id_bottom]
pressure_star = pressure[id_bottom]

# **** creating domain and boundary points **** #
fixed_bc = coords_test[:, 2] == 0  # fixed boundary
id_z0 = np.random.choice(np.where(fixed_bc)[0], 200, replace=False)
fixed_bc_points = coords_test[id_z0]

free_bc = coords_test[:, 2] == 1  # free boundary
id_z1 = np.random.choice(np.where(free_bc)[0], 200, replace=False)
free_bc_points = coords_test[id_z1]

id_top = np.random.choice(coords_boundary_top_test.shape[0], 1000, replace=False)  # top boundary
top_bc_points = coords_boundary_top_test[id_top]

boundary_random = np.vstack((coords_test[id_z0], coords_test[id_z1], top_bc_points, bottom_bc_points))
boundary_random_all = np.unique(boundary_random, axis=0)

# **** **** #
cloud_top_new = pv.PolyData(coords_boundary_top_test)
cloud_bottom_new = pv.PolyData(X_traction)

surface_top_new = cloud_top_new.delaunay_2d()
surface_bottom_new = cloud_bottom_new.delaunay_2d()

norms_top_new = surface_top_new.extract_surface()
norms_all_top_new = norms_top_new.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
normal_top = norms_all_top_new['Normals']
normal_top_random = normal_top[id_top]

norms_bottom_new = surface_bottom_new.extract_surface()
norms_all_bottom_new = norms_bottom_new.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
normal_bottom = norms_all_bottom_new['Normals']
normal_bottom_random = normal_bottom[id_bottom]

traction_data = np.hstack((X_traction, normal_bottom, pressure))
traction_bottom = np.hstack((bottom_bc_points, normal_bottom_random, pressure_star))
traction_top = np.hstack((top_bc_points, normal_top_random))

# Create a DataFrame for the coordinates
df_traction1 = pd.DataFrame(traction_bottom, columns=['x_traction', 'y_traction', 'z_traction', 'xn', 'yn', 'zn',
                                                    'pressure'])
df_traction2 = pd.DataFrame(traction_top, columns=['xb_t', 'yb_t', 'zb_t', 'xt', 'yt', 'zt'])
df_traction3 = pd.DataFrame(coords_domain, columns=['x', 'y', 'z'])
df_traction4 = pd.DataFrame(boundary_random_all, columns=['xb', 'yb', 'zb'])

df_traction1.to_csv('traction_bottom_rand.csv', index=False)
df_traction2.to_csv('traction_top_rand.csv', index=False)
df_traction3.to_csv('coords_rand.csv', index=False)
df_traction4.to_csv('boundary_rand.csv', index=False)

cloudd1 = pv.PolyData(fixed_bc_points)
cloudd2 = pv.PolyData(free_bc_points)
cloudd4 = pv.PolyData(top_bc_points)
cloudd5 = pv.PolyData(boundary_random_all)
cloudd6 = pv.PolyData(coords_domain)

plotter = pv.Plotter()
#plotter.add_points(cloudd1, color="black", point_size=3)
#plotter.add_points(cloudd2, color="black", point_size=3)
#plotter.add_points(cloudd3, color="black", point_size=3)
plotter.add_points(cloudd5, color="red", point_size=2)
plotter.add_points(cloud_domain, color="black", point_size=4)
plotter.enable_parallel_projection()
plotter.camera_position = 'xy'
#plotter.add_points(cloudd6, color="red", point_size=3)

plotter.show()


# **** for normals at top boundary **** #
norms_top = surface_top.extract_surface()
norms_all_top = norms_top.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
n_vectors_top = norms_all_top['Normals']

# **** for normals at bottom boundary **** #
norms_bottom = surface_bottom.extract_surface()
norms_all_bottom = norms_bottom.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
n_vectors_bottom = norms_all_bottom['Normals']

plotter2 = pv.Plotter()
plotter2.add_mesh(grid, style='surface', show_edges=True, color='white')
#plotter2.add_arrows(coords_boundary_top_test, normal_top, mag=0.004, color='red')  #
#plotter2.add_arrows(X_traction, normal_bottom, mag=0.004, color='green')  #
#plotter2.add_points(cloud_bottom_new, color="blue", point_size=5)
plotter2.add_arrows(top_bc_points, normal_top_random, mag=0.004, color='red')  #
plotter2.add_arrows(bottom_bc_points, normal_bottom_random, mag=0.004, color='green')  #
plotter2.enable_parallel_projection()
plotter2.camera_position = 'xy'
plotter2.show()


# **** for normals at face centers **** #
'''
norms = surface.extract_surface()
norms_all = norms.compute_normals(cell_normals=False, point_normals=True, auto_orient_normals=True)
n_vectors = norms_all['Normals']
n_points = norms_all.points
plotter3 = pv.Plotter()
plotter3.add_mesh(grid, style='surface', show_edges=True, color='white')
plotter3.add_arrows(n_points, n_vectors, mag=0.003, color='red')  #
plotter3.enable_parallel_projection()
plotter3.camera_position = 'xy'
# plotter3.show()
'''


# ## -------------------------------------------------------- #######

""" Save the coordinates of the normal vectors """
normal_vectors = np.hstack((coords_boundary_top, n_vectors_top, coords_boundary_bottom, n_vectors_bottom))

# Create a DataFrame for the coordinates
df_normals = pd.DataFrame(normal_vectors, columns=['cxt', 'cyt', 'czt', 'xt', 'yt', 'zt',
                                                   'cxb', 'cyb', 'czb', 'xb', 'yb', 'zb'])

# Save the coordinates to a CSV file
#df_normals.to_csv('normals.csv', index=False)

'''
# to visualize node labels

lab = np.zeros(len(coords_boundary_top[0:30, :]))
for i in range(len(coords_boundary_top[0:30, :])):
    lab[i] = i
labels = lab
actor = plotter.add_point_labels(
    coords_boundary_top[0:30, :],
    labels,
    italic=True,
    font_size=12,
    point_color='red',
    point_size=3,
    render_points_as_spheres=False,
    always_visible=True,
    shadow=False,
)

# plotter.show()
# Compute the total volume of the mesh
volume = surface.cells

'''
