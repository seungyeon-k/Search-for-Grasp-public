import numpy as np
import open3d as o3d
import torch

from functions.primitives import Superquadric
from functions.liegroup import skew_so3, expSO3
from functions.utils import quats_to_matrices

# color template
rgb_colors = np.array([
    [0, 0, 0], # black
    [100, 100, 100], # gray
    [255, 0, 0], # red
    [255, 96, 208], # pink
    [179, 173, 151], # brown
    [80, 208, 255],
    [0, 32, 255],
    [0, 192, 0],
    [255, 160, 16],
    [160, 128, 96],
])

def color_pc(labels_int):
    pc_colors = rgb_colors[labels_int.astype(np.int32).reshape(-1)]
    return pc_colors

# for segmentation
def color_pc_segmentation(labels):

    class_idx_label = np.argmax(labels, axis=1)
    pc_colors = rgb_colors[class_idx_label.astype(np.int32).reshape(-1)]
    
    return pc_colors

# for recognition
def sq_mesh_from_parameters(pose, parameters):

    # pose
    SE3 = pose.cpu().detach().numpy()
    
    # parameters 
    parameters = parameters.cpu().detach().numpy()

    # define mesh
    superquadric = Superquadric(
        SE3, 
        parameters, 
        color=[0.8, 0.8, 0.8], 
        resolution=5
    )
    mesh = superquadric.mesh
    
    return mesh

def sq_pc_from_parameters(pose, parameters, number_of_points=1024):
    mesh = sq_mesh_from_parameters(pose, parameters)
    pc = mesh.sample_points_uniformly(number_of_points=number_of_points)
    pc_numpy = np.asarray(pc.points)
    return pc_numpy

def make_table_pc(pc, scale=1.1, number_of_points=4096):
    
    # calculate min, max variables
    max_xyz = np.max(pc, axis=0) * scale
    min_xyz = np.min(pc, axis=0) * scale
    table_width = (max_xyz[0]-min_xyz[0])
    table_height = (max_xyz[1]-min_xyz[1])
    table_depth = 0.0001

    # table point cloud
    table = o3d.geometry.TriangleMesh.create_box(
        width=table_width, 
        height=table_height, 
        depth=table_depth
    )
    table.translate(min_xyz)
    pc_table = table.sample_points_uniformly(
        number_of_points=number_of_points
    )
    pc_table_numpy = np.asarray(pc_table.points)

    # color for table point cloud 
    colors_table = color_pc(
        4 * np.ones((len(pc_table_numpy)))
    )
    pc_table_with_colors = np.concatenate(
        (pc_table_numpy, colors_table), axis=1
    )

    return pc_table_with_colors
