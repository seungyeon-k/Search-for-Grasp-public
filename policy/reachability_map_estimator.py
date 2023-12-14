import torch

from copy import deepcopy
import open3d as o3d
from functions.superquadrics import sq_distance
from control.utils import (
	add_surrounding_objects, 
	get_shelf_sq_values,
)
from .utils import get_pc_afterimage
from .grasp_planner import superquadric_grasp_planner

############################################################
##################### CHECK FUNCTIONS ######################
############################################################

def check_graspability(
		target_object_pose,
		target_object_sq_param,
		Ts,
		parameters,
		pc_of_target_object,
		shelf_info=None,
		gripper_pc=None,
		visualize=False,
	):
	"""check whether the target object is graspable or not.

	Args:
		target_object_poses (4 x 4 torch tensor): target object's pose
		target_object_sq_params (5 torch tensor): target object's parameter
		Ts (n x 4 x 4 torch tensor): surrounding objects' poses
		parameters (n x 5 torch tensor): surrounding objects' parameters
		Ts_shelf (n_sh x 4 x 4, optional): shelf parts' poses.
		parameters_shelf (n_sh x 5, optional): shelf parts' parameters.
		gripper_pc (n_pc x 3, optional): gripper trajectory's point cloud.
		visualize (bool, optional): debugging mode

	Returns:
		score (bool): target object is graspable or not
  		valid_grasp_poses: (n_grasp x 4 x 4): valid (non-collided) grasp poses
	"""
	#print(target_object_pose, target_object_sq_param)
	# calculate grasp poses
	if (target_object_pose.ndim == 2) and (target_object_sq_param.ndim == 1):
		gripper_SE3s = superquadric_grasp_planner(
			target_object_pose, 
			target_object_sq_param
		)
	else:
		raise ValueError('Check the dimension of target object poses and sq parameters in graspability!')
	#print(gripper_SE3s.shape)
	device = Ts.device

	# check collision
	if len(gripper_SE3s) >= 1:
		pc_of_objects_afterimage = get_pc_afterimage(pc_of_target_object, distance = 0.2, directions = -gripper_SE3s[:,0:3,2])
		pc_of_objects_afterimage = torch.bmm(pc_of_objects_afterimage, gripper_SE3s[:,0:3,0:3]) - torch.bmm(gripper_SE3s[:,0:3,3].unsqueeze(1), gripper_SE3s[:,0:3,0:3])
		
		Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device)
		collision_gripper = check_collision(
			Ts,
			parameters,
			gripper_SE3s,
			gripper_pc,
			Ts_shelf=Ts_shelf,
			parameters_shelf=parameters_shelf,
			object_collision_scale=1.05,
			visualize=visualize,
			return_noncollide_poses=False
		) # n_grasp

		Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device, exclude_list=['middle'])
		collision_object = check_collision(
			Ts,
			parameters,
			gripper_SE3s,
			pc_of_objects_afterimage,
			Ts_shelf=Ts_shelf,
			parameters_shelf=parameters_shelf,
			object_collision_scale=1.05,
			visualize=visualize,
			return_noncollide_poses=False
		) # n_grasp

		collision = collision_gripper | collision_object
		valid_grasp_poses = gripper_SE3s[~collision]
		score = ~(collision.all())

	else:
		score = torch.tensor(False)
		valid_grasp_poses = torch.tensor([]).to(gripper_SE3s)

	return score, valid_grasp_poses

def check_reachability(
		gripper_SE3,
		Ts,
		parameters,
		Ts_shelf=None,
		parameters_shelf=None,
		gripper_pc=None,
		visualize=False,
		batch_size=10,
	):

	if gripper_pc is None:
		raise ValueError('Check gripper pc in reachability map estimator!')

	# check collision
	collision = []
	for Ts_batch, parameters_batch in zip(Ts.split(batch_size), parameters.split(batch_size)):
		collision_batch = check_collision(
			Ts_batch,
			parameters_batch,
			gripper_SE3,
			gripper_pc,
			Ts_shelf=Ts_shelf,
			parameters_shelf=parameters_shelf,
			object_collision_scale=1.05,
			visualize=visualize,
			return_noncollide_poses=False
		)
		collision.append(collision_batch)
	collision = torch.stack(collision)
	return ~collision.any(dim=0)

def check_collision(
		Ts, 
		parameters,
		target_SE3s, 
		target_pc, 
		Ts_shelf=None,
		parameters_shelf=None,
		visualize=False, 
		object_collision_scale=1.05,
		return_noncollide_poses=False,
		return_type='bool' # 'bool', 'sum'
	):
	"""_summary_

	Args:
		Ts (n x 4 x 4 torch tensor): surrounding objects' poses
		parameters (n x 5 torch tensor): surrounding objects' parameters
		target_SE3s (n_t x 4 x 4): targets' pose
		target_pc (n_pc x 3): target's point cloud
		Ts_shelf (n_sh x 4 x 4, optional): shelf parts' poses.
		parameters_shelf (n_sh x 5, optional): shelf parts' parameters.
		visualize (bool, optional): debugging mode
		object_collision_scale (float, optional)
		gripper_width (float, optional)
		return_noncollide_poses (bool, optional)

	Returns:
		scores (n_t bool torch tensor): scores for all targets
  		SE3s_noncollide (n_valid x 4 x 4 torch tensor): valid grasp poses (n_valid < n_t)
	"""
 
	# bigger
	parameters_big = deepcopy(parameters)
	parameters_big[:, 0] *= object_collision_scale
	parameters_big[:, 1] *= object_collision_scale
	parameters_big[:, 2] *= object_collision_scale

	# append table
	if (Ts_shelf is not None) and (parameters_shelf is not None):
		Ts_total = torch.cat([Ts, Ts_shelf], dim=0)
		parameters_total = torch.cat([parameters_big, parameters_shelf], dim=0)

	# match dimension
	if target_SE3s.ndim == 2:
		target_SE3s = target_SE3s.unsqueeze(0) # 1 x 4 x 4    		
	elif target_SE3s.ndim == 3: # target_SE3s dim: n_gr x 4 x 4
		pass
	else:
		raise NotImplementedError("Check the dimension of gripper SE3s.")

	if len(target_pc.shape) == 2:
		target_pc = target_pc.permute(1,0).unsqueeze(0) # 1 x 3 x n_pc
		# predefined grasp pose generation
		transformed_target_pc = (
			target_SE3s[:, :3, :3] @ target_pc 
			+ target_SE3s[:, :3, 3].unsqueeze(-1)
		).permute(0,2,1) # n_gr x n_pc x 3
	elif len(target_pc.shape) == 3:
		target_pc = target_pc.permute(0, 2, 1) # n_gr x 3 x n_pc
		transformed_target_pc = (
			torch.bmm(target_SE3s[:, :3, :3], target_pc) 
			+ target_SE3s[:, :3, 3].unsqueeze(-1)
		).permute(0,2,1) # n_gr x n_pc x 3

	# flatten
	n_gripper = transformed_target_pc.shape[0]
	n_pc = transformed_target_pc.shape[1]
	transformed_target_pc_flatten = transformed_target_pc.reshape(-1, 3)
	
	# calculate distance
	distances = sq_distance(
		transformed_target_pc_flatten, 
		Ts_total,
		parameters_total,
		mode='1'
	)
	
	distances = distances.reshape(n_gripper, n_pc, -1)
	distances_original = deepcopy(distances)
	
	distances = torch.min(distances, dim=-1)[0]
	distances = torch.min(distances, dim=-1)[0]

	# calculate score
	scores = torch.zeros_like(distances)
	scores[distances > 1] = 0
	scores[distances <= 1] = 1

	# return non-collide gripper poses
	SE3s_noncollide = target_SE3s[distances > 1]

	if visualize:
			
  		# superquadric meshes
		mesh_scene = []

		# draw objects
		mesh_scene += add_surrounding_objects(Ts_total, parameters_total)

		# draw gripper point cloud
		transformed_target_pc_numpy = transformed_target_pc.cpu().numpy()
		for i, pc in enumerate(transformed_target_pc_numpy):
			pc_o3d = o3d.geometry.PointCloud()
			pc_o3d.points = o3d.utility.Vector3dVector(pc)
			if torch.abs(scores[i] - 1) < 0.001:
				pc_o3d.paint_uniform_color([1, 0, 0])
			else:
				pc_o3d.paint_uniform_color([0, 1, 0])
			mesh_scene.append(pc_o3d)

		# visualize total mesh
		o3d.visualization.draw_geometries(mesh_scene)
  
	if return_type == 'sum':
		return distances_original.min(dim=1).values <= 1 #n_gripper, n_sq
	elif return_type == 'bool':
		if return_noncollide_poses:
			return scores == 1, SE3s_noncollide
		else:
			return scores == 1

############################################################
###################### GET FUNCTIONS #######################
############################################################

def get_graspability_hindrance(
	target_object_pose,
	target_object_sq_param,
	Ts,
	parameters,
	pc_of_target_object,
	shelf_info,
	gripper_pc=None,
	visualize=False,
	mode="smooth", 
):
	"""check the number of objects which hinder target object grasping.

	Args:
		target_object_pose (4 x 4 torch tensor): target object's pose
		target_object_sq_param (5 torch tensor): target object's parameter
		Ts (n x 4 x 4 torch tensor): surrounding objects' poses
		parameters (n x 5 torch tensor): surrounding objects' parameters
		pc_of_target_object (n_pc_t x 3) : target_object's pointcloud preresented in global frame
		Ts_shelf (n_sh x 4 x 4, optional): shelf parts' poses.
		parameters_shelf (n_sh x 5, optional): shelf parts' parameters.
		gripper_pc (n_pc x 3, optional): gripper trajectory's point cloud.
		visualize (bool, optional): debugging mode
		mode (str, optional): smooth / original : smooth - get smoothed value, original - get 0 or 1

	Returns:
		score (bool): target object is graspable or not
  		valid_grasp_poses: (n_grasp x 4 x 4): valid (non-collided) grasp poses
	"""

	# calculate grasp poses

	gripper_SE3s = superquadric_grasp_planner(
		target_object_pose, 
		target_object_sq_param
	)

	device = Ts.device

	# check collision
	if len(gripper_SE3s) >= 1:
		pc_of_objects_afterimage = get_pc_afterimage(pc_of_target_object, distance = 0.2, directions = -gripper_SE3s[:,0:3,2])
		pc_of_objects_afterimage = torch.bmm(pc_of_objects_afterimage, gripper_SE3s[:,0:3,0:3]) - torch.bmm(gripper_SE3s[:,0:3,3].unsqueeze(1), gripper_SE3s[:,0:3,0:3])

  
		Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device)
		collision_gripper = check_collision(
			Ts,
			parameters,
			gripper_SE3s,
			gripper_pc,
			Ts_shelf=Ts_shelf,
			parameters_shelf=parameters_shelf,
			object_collision_scale=1.05,
			visualize=visualize,
			return_noncollide_poses=False,
			return_type='sum',
		) # n_grasp x n_sq

		Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device, exclude_list=['middle'])
		collision_object = check_collision(
			Ts,
			parameters,
			gripper_SE3s,
			gripper_pc, # To do : change to pc_of_objects_afterimage
			Ts_shelf=Ts_shelf,
			parameters_shelf=parameters_shelf,
			object_collision_scale=1.05,
			visualize=visualize,
			return_noncollide_poses=False,
			return_type='sum',
		) # n_grasp x n_sq

		collision = collision_gripper[:,:len(Ts)] | collision_object[:,:len(Ts)]
		if mode == 'smooth':
			score = collision.sum(dim=1).min(dim=0).values
		elif mode == 'original':
			score = collision.max(dim=1).values.min(dim=0).values
		return score
	
	else:
		return 99.
	

def get_object_graspability(
		Ts, 
		parameters,
		pc_of_objects,
		gripper_open_pc,
		shelf_info, 
		visualize_graspability=False
	):

	# get graspability map
	graspability_list = []
	valid_grasp_poses_list = []
	for i in range(len(Ts)):
		graspability, valid_grasp_poses = check_graspability(
			Ts[i],
			parameters[i],
			Ts[torch.arange(len(Ts))!=i],
			parameters[torch.arange(len(parameters))!=i],
			pc_of_objects[i],
			shelf_info,
			gripper_pc=gripper_open_pc,
			visualize=visualize_graspability
		)

		graspability_list.append(graspability)
		valid_grasp_poses_list.append(valid_grasp_poses)

	return torch.tensor(graspability_list), valid_grasp_poses_list

# def get_reachability_map(
# 		grid, 
# 		Ts, 
# 		parameters, 
# 		gripper_open_pc, 
# 		shelf_info, 
# 		visualize_reachability=False
# 	):

# 	# get superquadric values of shelf
# 	device = Ts.device
# 	Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device, exclude_list='middle')

# 	# get gripper SE3s
# 	gripper_SE3s = torch.zeros(len(grid), 4, 4).to(device)
# 	gripper_SE3s[:, :3, 3] = grid

# 	# # get reachability map (for loop)
# 	# reachability_map = torch.zeros(len(grid))
# 	# for i in range(len(grid)):
# 	# 	reachability_map[i] = check_reachability(
# 	# 		gripper_SE3s[i],
# 	# 		Ts,
# 	# 		parameters,
# 	# 		Ts_shelf=Ts_shelf,
# 	# 		parameters_shelf=parameters_shelf,
# 	# 		gripper_pc=gripper_open_pc,
# 	# 		visualize=visualize_reachability
# 	# 	)

# 	# get reachability map (full batch)
# 	reachability_map = check_reachability(
# 		gripper_SE3s,
# 		Ts,
# 		parameters,
# 		Ts_shelf=Ts_shelf,
# 		parameters_shelf=parameters_shelf,
# 		gripper_pc=gripper_open_pc,
# 		visualize=visualize_reachability
# 	)

# 	return reachability_map

def get_reachability_map(
		grid,
		grid_shape,
		Ts, 
		parameters, 
		gripper_open_pc, 
		shelf_info, 
		visualize_reachability=False
	):

	# get superquadric values of shelf
	device = Ts.device
	Ts_shelf, parameters_shelf = get_shelf_sq_values(shelf_info, device=device, exclude_list=['middle'])

	# get gripper SE3s
	gripper_SE3s = torch.eye(4).repeat(len(grid), 1, 1).to(device)
	gripper_SE3s[:, :3, 3] = grid

	# get reachability map (full batch)
	reachability_map = check_reachability(
		gripper_SE3s,
		Ts,
		parameters,
		Ts_shelf=Ts_shelf,
		parameters_shelf=parameters_shelf,
		gripper_pc=gripper_open_pc,
		visualize=visualize_reachability
	)
 
	reachability_map = reachability_map.reshape(grid_shape)
	reachability_map = (~reachability_map).cumsum(dim=1) == 0
	reachability_map = reachability_map.reshape(-1)
	return reachability_map