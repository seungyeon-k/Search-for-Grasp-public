import torch
import open3d as o3d
import time

from .reachability_map_estimator import get_object_graspability, get_reachability_map
from .utils import get_pc_afterimage

def get_random_graspable_object_and_pose(
	objects_poses,
	objects_sq_params,
	pc_of_objects,
	pc_of_gripper,
	shelf_info,
	remove_idxs,
):
	"""randomly select an object and grasping pose from observed non-target objects. note that sampling is restricted to graspable objects and valid grasping pose.

	Args:
		other_objects_poses (n x 4 x 4 tensor): SE(3) pose matrices of observed non-target objects
		other_objects_sq_params (n x 5 tensor): superquadric parameters of observed non-target objects
		remove_idxs (list of integers): list of objects index which we want not to grasp (specifically, the object which has been manipulated before)

	Returns:
		random_object (int) : index of selected object
		random_grasping_pose (4 x 4 tensor) : SE(3) pose matrices of selected grasping pose
	"""
	
	# get graspability of all objects
	graspability, grasping_poses = get_object_graspability(
		objects_poses, 
		objects_sq_params,
		pc_of_objects,
		pc_of_gripper,
		shelf_info,
		visualize_graspability=False
	)
	
	# do not manipulate the objects in the remove_idxs
	graspability[[remove_idxs]] = False

	objects_idx = torch.arange(len(objects_poses)).to(objects_poses.device)
	valid_objects_idx = objects_idx[graspability]
	valid_grasping_poses = [grasping_poses[i] for i in valid_objects_idx]
		
	return valid_objects_idx, valid_grasping_poses

def place_object_at_random_position(
	objects_poses,
	objects_sq_params,
	pc_of_objects,
	pc_of_gripper,
	grasped_object_idx,
	grasping_pose,
	placing_grid,
	placing_grid_shape,
	shelf_info
):

	# concatenate object pc and gripper pc with sampled pose                        
	pc_of_gripper = (grasping_pose[:3, :3] @ pc_of_gripper.T + grasping_pose[:3, 3].unsqueeze(-1)).T
	pc_of_object = get_pc_afterimage(pc_of_objects[grasped_object_idx], directions = -grasping_pose[:3, 2].unsqueeze(0)).squeeze()
	pc_grasper_with_object = torch.cat([pc_of_object, pc_of_gripper], dim=0)
	
	# rotate whole pointcloud so that object remains in the same position and the gripper faces front side.
	new_pose = torch.zeros(4, 4).to(objects_poses.device)
	RotZ_angle = torch.atan2(grasping_pose[1, 2], grasping_pose[0, 2])
	RotZ = torch.tensor([[torch.cos(RotZ_angle), -torch.sin(RotZ_angle), 0.],
						 [torch.sin(RotZ_angle),  torch.cos(RotZ_angle), 0.],
						 [0.,                     0.,                    1.]]).T.to(objects_poses.device)
	new_pose[0:3, 0:3] = RotZ
	new_pose[0:3, 3] = (torch.eye(3).to(objects_poses.device) - new_pose[0:3, 0:3]) @ objects_poses[grasped_object_idx][:3, 3]
	
	pc_grasper_with_object = (new_pose[:3, :3] @ pc_grasper_with_object.T + new_pose[:3, 3].unsqueeze(-1)).T
	
	# find valid position to place the object

	reachable_position_bools = get_reachability_map(
		placing_grid,
		placing_grid_shape,
		objects_poses, 
		objects_sq_params,
		pc_grasper_with_object - objects_poses[grasped_object_idx][:3, 3], 
		shelf_info,
		visualize_reachability=False
	)

	# random sample the object to be placed
	reachable_position = placing_grid[reachable_position_bools]
	if len(reachable_position) == 0:
		return None
	random_position = reachable_position[torch.randint(low=0, high=len(reachable_position), size=[])]

	# update objects pose
	placed_pose = torch.eye(4).to(objects_poses.device)
	placed_pose[0:3, 0:3] = new_pose[0:3, 0:3] @ objects_poses[grasped_object_idx][0:3, 0:3]
	placed_pose[0:3, 3] = random_position

	return placed_pose
		