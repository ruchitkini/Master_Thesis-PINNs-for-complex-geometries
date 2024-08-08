import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# ############# GET ABAQUS MESH ############## #
node_df = pd.read_csv('results/Nodes.csv')
element_df = pd.read_csv('results/Elements.csv')
node_set_df = pd.read_csv('results/Node_Set.csv')
element_set_df = pd.read_csv('results/Element_Set.csv')

node_ids = node_df['Node_ID']  # node ids
element_ids = element_df['Element_ID']  # element ids

node_set_df.fillna(0, inplace=True)  # fill empty values
node_set_ids = np.concatenate(node_set_df.values, axis=0)  # concatenate rows
node_set_ids = node_set_ids[node_set_ids != 0]  # delete 0 values
node_set_ids = node_set_ids.astype(int)  # node set ids

element_set_df.fillna(0, inplace=True)  # fill empty values
element_set_ids = np.concatenate(element_set_df.values, axis=0)  # concatenate rows
element_set_ids = element_set_ids[element_set_ids != 0]  # delete 0 values
element_set_ids = element_set_ids.astype(int)  # element set ids

XP_fem = np.zeros((len(node_df), 3))  # points array
XP_fem[:, 0], XP_fem[:, 1], XP_fem[:, 2] = node_df['X'], node_df['Y'], node_df['Z']
XP_fem_2D = np.zeros((len(node_df), 2))
XP_fem_2D[:, 0], XP_fem_2D[:, 1] = node_df['X'], node_df['Z']

Faces_fem = np.zeros((len(element_df), 8))  # faces
(Faces_fem[:, 0], Faces_fem[:, 1], Faces_fem[:, 2], Faces_fem[:, 3],
 Faces_fem[:, 4], Faces_fem[:, 5], Faces_fem[:, 6], Faces_fem[:, 7]) = (element_df['n1'], element_df['n2'],
                                                                        element_df['n3'], element_df['n4'],
                                                                        element_df['n5'], element_df['n6'],
                                                                        element_df['n7'], element_df['n8'])
set_points = XP_fem[node_set_ids-1]
set_elements = Faces_fem[element_set_ids-1]
set_faces = [tuple(int(num) for num in sublist) for sublist in set_elements]
akdfjalk = set_faces[0]

new_set_faces = []

for face in set_faces:
    # Initialize a list to hold the nodes of the new face
    new_face_nodes = []
    # Check if all node IDs in the face are present in node_set_ids
    for node_id in face:
        if node_id in node_set_ids:
            # If the node ID is in node_set_ids, add it to the new face nodes
            new_face_nodes.append(node_id)
    # Convert the list of new face nodes to a tuple
    new_face = tuple(new_face_nodes)
    # Append the new face to the list of new set faces
    if len(new_face) > 0:
        new_set_faces.append(new_face)

# Convert the list of new faces to a tuple of tuples
new_set_faces2 = np.array(new_set_faces)
new_set_faces = tuple(new_set_faces)

front_left_vertices = XP_fem[new_set_faces2[:, 0]-1, :]
back_left_vertices = XP_fem[new_set_faces2[:, 1]-1, :]
back_right_vertices = XP_fem[new_set_faces2[:, 3]-1, :]
front_right_vertices = XP_fem[new_set_faces2[:, 2]-1, :]

Fine_centroid_x = 0.25 * (front_left_vertices[:, 0] + back_left_vertices[:, 0] + back_right_vertices[:, 0] + front_right_vertices[:, 0])
Fine_centroid_y = 0.25 * (front_left_vertices[:, 2] + back_left_vertices[:, 2] + back_right_vertices[:, 2] + front_right_vertices[:, 2])
Fine_centroid_z = 0.25 * (front_left_vertices[:, 1] + back_left_vertices[:, 1] + back_right_vertices[:, 1] + front_right_vertices[:, 1])

def plot_3D_fem_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot vertices
    # ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='b')

    # Plot faces
    for face in faces:
        v0 = vertices[face[0]-1]
        v1 = vertices[face[1]-1]
        v2 = vertices[face[2]-1]
        v3 = vertices[face[3]-1]
        # v4 = vertices[face[4]-1]
        # v5 = vertices[face[5]-1]
        # v6 = vertices[face[6]-1]
        # v7 = vertices[face[7]-1]
        verts = [v0, v1, v3, v2, v0]
        xs, ys, zs = zip(*verts)
        ax.plot(xs, ys, zs, color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Mesh')

    # Set limits for x, y, and z axes
    ax.set_xlim(0, 1)  # X axis limits
    ax.set_ylim(-0.3, 0.3)  # Y axis limits
    ax.set_zlim(0, 1)  # Z axis limits

    # Set initial viewing angle to xz plane
    ax.view_init(elev=0, azim=90)

    plt.show()


def plot_2D_fem_mesh(vertices, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(Fine_centroid_x, Fine_centroid_y, s=1, color='black')

    # Plot faces
    for face in faces:
        v0 = vertices[face[0]-1]
        v1 = vertices[face[1]-1]
        v2 = vertices[face[2]-1]
        v3 = vertices[face[3]-1]

        verts = [v0, v1, v3, v2, v0]
        xs, ys = zip(*verts)
        ax.plot(xs, ys, color='k')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.axis("equal")
    ax.set_title('2D Quad Mesh')

    plt.show()


# Plot the mesh
Faces_fem = [tuple(int(num) for num in sublist) for sublist in Faces_fem]
# plot_3D_fem_mesh(XP_fem, new_set_faces)
# plot_2D_fem_mesh(XP_fem_2D, new_set_faces)

# ######### AEROSANDBOX Mesh ############# #
df = pd.read_csv("pressure_distribution_coarse.csv")

xp, yp, zp = df['xp'], df['yp'], df['zp']
f1, f2, f3, f4 = df['f1'], df['f2'], df['f3'], df['f4']
xf, yf, zf = df['xf'], df['yf'], df['zf']
pressure = df['pressure'].tolist()

XP = np.zeros((len(xp), 3))  # points array
XP[:, 0], XP[:, 1], XP[:, 2] = xp, yp, zp
XP_2D = np.zeros((len(xp), 2))  # points array
XP_2D[:, 0], XP_2D[:, 1] = xp, yp

Faces = np.zeros((len(f1), 4))  # faces
Faces[:, 0], Faces[:, 1], Faces[:, 2], Faces[:, 3] = f1, f2, f3, f4
Frows_with_nan = np.isnan(Faces).any(axis=1)
Faces = Faces[~Frows_with_nan]
Faces = [tuple(int(num) for num in sublist) for sublist in Faces]

Face_centers = np.column_stack((xf, yf, zf))
fc_mask = ~np.isnan(Face_centers).any(axis=1)
Face_centers = Face_centers[fc_mask]

Pressure = [x for x in pressure if not np.isnan(x)]


def plot_2D_aero_mesh_with_pressure(vertices, faces, pressure_values):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Plot vertices
    ax.scatter(Face_centers[:, 0], Face_centers[:, 1], s=5, c=pressure_values, cmap='viridis')

    # Plot faces
    for face in faces:
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        v3 = vertices[face[3]]

        verts = [v0, v1, v2, v3, v0]
        xs, ys = zip(*verts)
        ax.plot(xs, ys, color='grey')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.axis("equal")
    ax.set_title('Mesh with Pressure Values')

    # Add color bar
    cbar = plt.colorbar(ax.scatter([], [], [], c=[], cmap='viridis'))
    cbar.set_label('Pressure')

    plt.show()


# Plot
plot_2D_aero_mesh_with_pressure(XP_2D, Faces, Pressure)


# ############### MAPPING PRESSURE FROM AERO MESH TO ABAQUS MESH ################ #
vertices_coarse = np.array(XP[:, 0:2])
elements_coarse = Faces

# Define vertices (nodes) for fine mesh
all_vertices_fine = np.array(XP_fem[:, 0:3])
vertices_fine = np.array(set_points[:, 0:3])

# Define element connectivity (quad elements) for fine mesh
elements_fine = new_set_faces
# elements_fine = [tuple(int(num) for num in sublist) for sublist in elements_fine]

# Define arbitrary pressure values for coarse mesh elements
pressure_values_coarse = Pressure  # Example pressure value for coarse element

# Calculate the pressure value for each fine mesh element based on the coarse element it fits into
pressure_values_fine = np.zeros(len(elements_fine))
N_fine = []


for i, element_coarse in enumerate(elements_coarse):
    # Calculate centroid of coarse element
    centroid_x_coarse = np.mean([vertices_coarse[node][0] for node in element_coarse])
    centroid_y_coarse = np.mean([vertices_coarse[node][1] for node in element_coarse])

    cc_x = Face_centers[i][0]
    cc_y = Face_centers[i][1]

    # Count how many fine elements fit into this coarse element
    num_fine_elements = 0
    for j, element_fine in enumerate(elements_fine):
        # Calculate centroid of fine element
        # hghf = vertices_fine[node_set_ids][0]
        centroid_x_fine = np.mean([all_vertices_fine[node-1][0] for node in element_fine])
        centroid_y_fine = np.mean([all_vertices_fine[node-1][2] for node in element_fine])

        cx_fine = Fine_centroid_x[j]
        cy_fine = Fine_centroid_y[j]

        # Check if fine element centroid is inside coarse element
        xmin = np.min([vertices_coarse[node][0] for node in element_coarse])
        xmax = np.max([vertices_coarse[node][0] for node in element_coarse])
        ymin = np.min([vertices_coarse[node][1] for node in element_coarse])
        ymax = np.max([vertices_coarse[node][1] for node in element_coarse])
        if (xmin <= centroid_x_fine <= xmax) and (ymin <= centroid_y_fine <= ymax):
            num_fine_elements += 1
            # n_fine_elements = num_fine_elements
            pressure_values_fine[j] += pressure_values_coarse[i]

    n_fine_elements = np.zeros(num_fine_elements)
    for k in range(num_fine_elements):
        n_fine_elements[k] = num_fine_elements
    N_fine.extend(n_fine_elements)

print("appended N_fine = ", N_fine)


# Divide pressure values by the number of fine elements each coarse element contains
pressure_values_fine = pressure_values_fine
n_counts = np.zeros((len(pressure_values_fine)))
i = 0
for value in pressure_values_fine:
    count = np.count_nonzero(pressure_values_fine == value)
    n_counts[i] = count
    i = i+1
    print()

new_pressure_values_fine = pressure_values_fine/n_counts
print("count = ", n_counts)
print(new_pressure_values_fine)

df_p = pd.DataFrame(new_pressure_values_fine)
# df_p.to_csv("fine_pressure_abaqus.csv", index=False)

x_traction, y_traction, z_traction = Fine_centroid_x, Fine_centroid_z, Fine_centroid_y
traction_data = np.column_stack((x_traction, y_traction, z_traction, new_pressure_values_fine))
df_traction = pd.DataFrame(traction_data, columns=['x_traction', 'y_traction', 'z_traction', 'pressure'])
df_traction.to_csv("traction_data.csv", index=False)

# pressure_values_fine = pressure_values_fine/N_fine
# pressure_values_fine = pressure_values_mapped

print(" coarse pressure was = ", pressure_values_coarse)
print(" new fine pressure is = ", pressure_values_fine)


# ############## comparison scatter plot ############### #
centroid_x_fine_values = []
centroid_y_fine_values = []

for j, element_fine in enumerate(elements_fine):
    centroid_x_fine = np.mean([all_vertices_fine[node-1][0] for node in element_fine])
    centroid_y_fine = np.mean([all_vertices_fine[node-1][2] for node in element_fine])

    centroid_x_fine_values.append(centroid_x_fine)
    centroid_y_fine_values.append(centroid_y_fine)


# centroid_x_fine_values = np.mean

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Scatter plot 1
#scatter01 = axes[0].scatter(vertices_coarse[:, 0], vertices_coarse[:, 1], s=1, color='black')
scatter1 = axes[0].scatter(Face_centers[:, 0], Face_centers[:, 1], s=3, c=pressure_values_coarse, label='Aero Pressure')
axes[0].set_title('Plot 1')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend()

# Add color bar for plot 1
cbar1 = fig.colorbar(scatter1, ax=axes[0])
cbar1.set_label('Pressure')

# Scatter plot 2
#scatter02 = axes[1].scatter(vertices_fine[:, 0], vertices_fine[:, 2], s=0.5, color='black')
scatter2 = axes[1].scatter(centroid_x_fine_values, centroid_y_fine_values, s=3, c=new_pressure_values_fine, label='Abaqus Pressure')
axes[1].set_title('Plot 2')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
axes[1].legend()

# Add color bar for plot 2
cbar2 = fig.colorbar(scatter2, ax=axes[1])
cbar2.set_label('Pressure')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


