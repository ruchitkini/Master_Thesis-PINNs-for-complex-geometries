""" Creating different sets of elements to apply traction on faces --> in abaqus """

import numpy as np
import pandas as pd

ele_S5_df = pd.read_csv('results/ele_S5.csv')
ele_S3_df = pd.read_csv('results/ele_S3.csv')
ele_S4_df = pd.read_csv('results/ele_S4.csv')
ele_S6_df = pd.read_csv('results/ele_S6.csv')

ele_S5_df.fillna(0, inplace=True)
s5 = ele_S5_df.values
s5 = s5[s5 != 0]  # delete 0 values
s5 = s5.flatten()[:, None]
tuple_s5 = tuple(int(x[0]) for x in s5)

ele_S3_df.fillna(0, inplace=True)
s3 = ele_S3_df.values
s3 = s3[s3 != 0]  # delete 0 values
s3 = s3.flatten()[:, None]
tuple_s3 = tuple(int(x[0]) for x in s3)

ele_S4_df.fillna(0, inplace=True)
s4 = ele_S4_df.values
s4 = s4[s4 != 0]  # delete 0 values
s4 = s4.flatten()[:, None]
tuple_s4 = tuple(int(x[0]) for x in s4)

ele_S6_df.fillna(0, inplace=True)
s6 = ele_S6_df.values
s6 = s6[s6 != 0]  # delete 0 values
s6 = s6.flatten()[:, None]
tuple_s6 = tuple(int(x[0]) for x in s6)


# write to csv
df_s5 = pd.DataFrame(tuple_s5)
df_s5.to_csv("element_s5.csv", index=False)

df_s3 = pd.DataFrame(tuple_s3)
df_s3.to_csv("element_s3.csv", index=False)

df_s4 = pd.DataFrame(tuple_s4)
df_s4.to_csv("element_s4.csv", index=False)

df_s6 = pd.DataFrame(tuple_s6)
df_s6.to_csv("element_s6.csv", index=False)


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
