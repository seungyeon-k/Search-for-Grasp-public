import numpy as np
import torch
from copy import deepcopy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from datetime import datetime
from data_generation.utils import label_to_color
from functions.primitives import Superquadric
from functions.lie import quats_to_matrices, get_SE3s

############################################################
###################### ADD FUNCTIONS #######################
############################################################

def add_pc(pc):

	# list
	mesh_list = []

	# point cloud
	pc_o3d = o3d.geometry.PointCloud()
	pc_o3d.points = o3d.utility.Vector3dVector(pc.T)
	mesh_list.append(pc_o3d)

	return mesh_list

def add_surrounding_objects(Ts, parameters, colors=None):
	
	# list
	mesh_list = []	

	# torch to numpy
	Ts_numpy = Ts.cpu().numpy()
	parameters_numpy = parameters.cpu().numpy()

	# add geometries and lighting
	for id, (SE3, parameter) in enumerate(zip(Ts_numpy, parameters_numpy)):
		
		# get mesh
		mesh = Superquadric(SE3, parameter, resolution=20).mesh
		mesh.compute_vertex_normals()
		if colors is None:
			mesh.paint_uniform_color(1 * np.random.rand(3))
		else:
			if colors.ndim == 1:
				mesh.paint_uniform_color(colors)	
			elif colors.ndim == 2:
				mesh.paint_uniform_color(colors[id])
		
		# mesh 
		mesh_list.append(mesh)

	return mesh_list

def add_shelf(shelf, exclude_list=[]):
	
	# list
	mesh_list = []

	# get shelf parts
	part_keys = [key for key in shelf.keys() if key.startswith('shelf')]

	# get position and orientation
	global_position = shelf['global_position']
	global_orientation = shelf['global_orientation']
	global_SO3 = quats_to_matrices(global_orientation)
	global_SE3 = get_SE3s(global_SO3, global_position)

	for part_key in part_keys:
			
		# exclude
		if part_key.split('_')[-1] in exclude_list:
			continue

		# load info	
		size = shelf[part_key]['size']
		position = shelf[part_key]['position']
		
		# get open3d mesh
		mesh = o3d.geometry.TriangleMesh.create_box(
			width = size[0], 
			height = size[1], 
			depth = size[2]
		)
		mesh.translate([-size[0]/2, -size[1]/2, -size[2]/2])
		mesh.compute_vertex_normals()

		# transform
		mesh.translate(position)
		mesh.transform(global_SE3)
		mesh.paint_uniform_color([101/255/2, 67/255/2, 33/255/2])

		# append
		mesh_list.append(mesh)	

	return mesh_list

def add_map(grid, map, shelf):
	
	# list
	mesh_list = []	

	# numpy
	grid_numpy = grid.cpu().numpy()
	map_numpy = map.cpu().numpy().astype(np.int32)

	# make color
	rgb = np.array([
		[153, 0, 17], # red
		[111, 146, 110], # green
	]) / 255.0
	color = rgb[map_numpy]

	# make point cloud 
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(grid_numpy)
	pcd.colors = o3d.utility.Vector3dVector(color)	

	# mesh 
	mesh_list.append(pcd)

	return mesh_list

############################################################
######################## RENDERING #########################
############################################################

def render_segmentation(pc, labels):
		
	color_pred = label_to_color(labels)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	pcd.colors = o3d.utility.Vector3dVector(color_pred)		
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)	
	o3d.visualization.draw_geometries([pcd, frame])

def render_objects( 
		Ts, 
		parameters,
		shelf=None, 
		resolution=10
	):

	# list
	mesh_list = []
 
	# add geometries and lighting
	mesh_list += add_surrounding_objects(Ts, parameters)

	# add shelf
	if shelf is not None:
		mesh_list += add_shelf(shelf)

	o3d.visualization.draw_geometries(mesh_list)

def render_objects_with_pc(
		pc, 
		Ts, 
		parameters,
		shelf=None, 
		resolution=10
	):

	# list
	mesh_list = []

	# point cloud
	mesh_list += add_pc(pc)

	# add geometries and lighting
	mesh_list += add_surrounding_objects(Ts, parameters)

	# add shelf
	if shelf is not None:
		mesh_list += add_shelf(shelf)

	o3d.visualization.draw_geometries(mesh_list)

def render_map(
		Ts, 
		parameters,
		grid,
		map, 
		shelf=None,
		resolution=10
	):
	
	# list
	mesh_list = []

	# add geometries and lighting
	mesh_list += add_surrounding_objects(Ts, parameters)

	# add reachability map
	mesh_list += add_map(grid, map, shelf)

	# add shelf
	if shelf is not None:
		mesh_list += add_shelf(shelf, exclude_list=['upper', 'middle'])

	# visualize
	o3d.visualization.draw_geometries(mesh_list)

def render_graspability(
		Ts, 
		parameters,
		graspability,
		shelf=None,
		resolution=10
	):
	
	# list
	mesh_list = []

	# numpy
	graspability_numpy = graspability.cpu().numpy().astype(np.int32)

	# make color
	rgb = np.array([
	 	[153, 0, 17], # red
		[111, 146, 110], # green
	]) / 255.0
	colors = rgb[graspability_numpy]

	# add geometries and lighting
	mesh_list += add_surrounding_objects(Ts, parameters, colors=colors)

	# add shelf
	if shelf is not None:
		mesh_list += add_shelf(shelf, exclude_list=['upper'])

	# visualize
	o3d.visualization.draw_geometries(mesh_list)


#############################################################
######################### ALGORITHM #########################
#############################################################

def get_shelf_bounds(shelf, grid_size='large'):

	# get position and orientation
	global_position = shelf['global_position']
	global_orientation = shelf['global_orientation']
	global_SO3 = quats_to_matrices(global_orientation)
	global_SE3 = get_SE3s(global_SO3, global_position)

	# load info	
	size = shelf['shelf_middle']['size']
	position = shelf['shelf_middle']['position']
	
	# get open3d mesh
	mesh = o3d.geometry.TriangleMesh.create_box(
		width = size[0], 
		height = size[1], 
		depth = size[2]
	)
	mesh.translate([-size[0]/2, -size[1]/2, -size[2]/2])
	mesh.compute_vertex_normals()

	# transform
	mesh.translate(position)
	mesh.transform(global_SE3)
	vertices_numpy = np.asarray(mesh.vertices)

	if grid_size == 'large':
		return np.min(vertices_numpy, axis=0) + np.array([0.05, 0.05, 0]), np.max(vertices_numpy, axis=0) - np.array([0.05, 0.05, 0])
	elif grid_size == 'small':
		return np.min(vertices_numpy, axis=0) + np.array([0.05, 0.18, 0]), np.max(vertices_numpy, axis=0) - np.array([0.05, 0.28, 0])

def get_shelf_grid(shelf, resolution=0.04, dtype='numpy', grid_dim='2D', grid_size='large'):

	# get shelf bounds
	shelf_bounds_min, shelf_bounds_max = get_shelf_bounds(shelf, grid_size)

	# shelf height
	shelf_height = 0.3

	# get the number of grids
	size_x = np.round(
		(shelf_bounds_max[0] - shelf_bounds_min[0]) / resolution
	).astype(int)
	size_y = np.round(
		(shelf_bounds_max[1] - shelf_bounds_min[1]) / resolution
	).astype(int)
	if grid_dim == '3D':
		size_z = np.round(
			shelf_height / resolution
		).astype(int)
	
	# meshgrid
	x = np.linspace(shelf_bounds_min[0], shelf_bounds_max[0], size_x)
	y = np.linspace(shelf_bounds_min[1], shelf_bounds_max[1], size_y)
	if grid_dim == '2D':
		X, Y = np.meshgrid(x, y)
	elif grid_dim == '3D':
		z = np.linspace(shelf_bounds_max[2], shelf_bounds_max[2] + shelf_height, size_z)
		X, Y, Z = np.meshgrid(x, y, z)
	elif grid_dim == 'SE2':
		t = np.linspace(0, 1, 7)[:-1] * np.pi
		X, Y, T = np.meshgrid(x, y, t)
	
	# reshape grids
	X = X.reshape(-1, 1)
	Y = Y.reshape(-1, 1)
	if grid_dim == '2D':
		Z = np.array([[shelf_bounds_max[2]]]).repeat(len(X), 0)
		grid = np.concatenate((X, Y, Z), axis=1)
	elif grid_dim == '3D':
		Z = Z.reshape(-1, 1)
		grid = np.concatenate((X, Y, Z), axis=1)
	elif grid_dim == 'SE2':
		T = T.reshape(-1, 1)
		Z = np.array([[shelf_bounds_max[2]]]).repeat(len(X), 0)
		grid = np.concatenate((X, Y, Z, T), axis=1)
	

	if dtype == 'numpy':
		return grid, [size_x, size_y]
	elif dtype == 'torch':
		return torch.tensor(grid).float(), [size_x, size_y]

def get_shelf_sq_values(shelf, exclude_list=[], device=None):
		
	# list
	mesh_list = []

	# get shelf parts
	part_keys = [key for key in shelf.keys() if key.startswith('shelf')]

	# get position and orientation
	global_position = shelf['global_position']
	global_orientation = shelf['global_orientation']
	global_SO3 = quats_to_matrices(global_orientation)

	# initialize
	Ts_shelf = []
	parameters_shelf = []

	for part_key in part_keys:
			
		# exclude
		if part_key.split('_')[-1] in exclude_list:
			continue

		# load info	
		size = shelf[part_key]['size']
		position = shelf[part_key]['position']
		
		# transform
		part_SO3 = global_SO3
		part_position = global_position + global_SO3.dot(position)
		T = get_SE3s(part_SO3, part_position)
		T = torch.tensor(T).float()

		# parameters
		parameter = torch.tensor([size[0]/2, size[1]/2, size[2]/2, 0.2, 0.2])

		# append
		Ts_shelf.append(T)
		parameters_shelf.append(parameter)	

	Ts_shelf = torch.stack(Ts_shelf)
	parameters_shelf = torch.stack(parameters_shelf)

	Ts_shelf = Ts_shelf.to(device)
	parameters_shelf = parameters_shelf.to(device)

	return Ts_shelf, parameters_shelf
