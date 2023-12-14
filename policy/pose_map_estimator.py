import numpy as np
import torch
from functions.utils_wandb import sq_pc_from_parameters
from functions.superquadrics import sq_distance
from control.utils import (
	add_surrounding_objects, 
	add_shelf,
	get_shelf_sq_values,
)
from .utils import get_poses_from_grid
import matplotlib.pyplot as plt
from copy import deepcopy

def check_depth_feasibility(
        target_object_poses,
        target_object_sq_params,
        other_objects_poses,
        other_objects_sq_params,
        depth_renderer, 
        thld=0.001
        ):
    """
    check if the target objects change the original depth image composed of other objects.
    
    input: target_object_poses (n x 4 x 4 torch tensor)
            target_object_sq_params (n x 5 torch tensor)
            other_objects_poses (m x x 4 x 4 torch tensor)
            other_objects_sq_params (m x 5 torch tensor)
    output: booleans (n)
    """
    original_depth_image = depth_renderer.depth_render(other_objects_poses,
                                                       other_objects_sq_params)
    depth_image = depth_renderer.depth_render_batchwise(target_object_poses,
                                                        target_object_sq_params,
                                                        other_objects_poses,
                                                        other_objects_sq_params)

    return (depth_image - original_depth_image).norm(dim=[1,2]) < thld

def check_collision(
        target_object_poses,
        target_object_sq_params,
        other_objects_poses,
        other_objects_sq_params,
        batch_size=10
    ):
    """
    check if target objects collide with other objects.
    
    input: target_object_poses (n x 4 x 4 torch tensor)
            target_object_sq_params (n x 5 torch tensor)
            other_objects_poses (m x x 4 x 4 torch tensor)
            other_objects_sq_params (m x 5 torch tensor)
    output: booleans (n)
    """
    # get pointcloud of other objects
    other_objects_point_clouds = np.concatenate(
        [
            sq_pc_from_parameters(other_objects_pose, other_objects_sq_param)
            for other_objects_pose, other_objects_sq_param
            in
            zip(other_objects_poses, other_objects_sq_params)
        ]
        ,axis = 0
        )
    other_objects_point_clouds = torch.from_numpy(other_objects_point_clouds).float().to(other_objects_sq_params)
    
    sq_value_list = []
    # get target objects' superquadric function values on other objects' point clouds. if the point is on the surface of target object, the value is 1.
    for target_object_pose_batch, target_object_sq_param_batch in zip(target_object_poses.split(batch_size), target_object_sq_params.split(batch_size)):
        sq_value_batch = sq_distance(other_objects_point_clouds, target_object_pose_batch, target_object_sq_param_batch)
        sq_value_list.append(sq_value_batch)
    sq_value = torch.cat(sq_value_list, dim=1)
    # if sq_value < 1, it means some of other objects' points are in the target objects, i.e. collision occurs.
    boolean = (sq_value < 1).any(dim=0)
    return boolean

def check_pose(
        target_object_poses,
        target_object_sq_param, 
        other_objects_poses,
        other_objects_sq_params, 
        depth_renderer
    ):
    """
    check if the target objects are feasible in the sense that depth images are not changed and target objects don't collide with other objects. 
    
    input: target_object_poses (n x 4 x 4 torch tensor)
           target_object_sq_param (5-dim torch tensor)
           other_objects_poses (m x 4 x 4 torch tensor)
           other_objects_sq_params (m x 5 torch tensor)
    output: booleans (n)
    """
    target_poses_num = target_object_poses.shape[0]
    target_object_sq_param = target_object_sq_param.repeat(target_poses_num, 1)
    
    # check depth image
    depth = check_depth_feasibility(
        target_object_poses,
        target_object_sq_param,
        other_objects_poses,
        other_objects_sq_params, 
        depth_renderer
        )

    # check collision
    collsion = check_collision(
        target_object_poses,
        target_object_sq_param,
        other_objects_poses,
        other_objects_sq_params
        )

    return depth & ~collsion

def get_pose_map(
        grid,
	    target_object_sq_param,
		other_objects_poses,
		other_objects_sq_params,
        depth_renderer,
        object_scale=1.05,
		):
    """
    check if the target objects are feasible for each position in the grid 
    
    input: grid (n x 3 torch tensor)
           target_object_sq_param (5-dim torch tensor)
           other_objects_poses (m x 4 x 4 torch tensor)
           other_objects_sq_params (m x 5 torch tensor)
    output: booleans (n)
    """
    
	# make SE(3) matrix from position grid
    target_object_poses = get_poses_from_grid(grid, target_object_sq_param[2])
    
    temp_other_objects_sq_params = deepcopy(other_objects_sq_params)
    temp_other_objects_sq_params[:,0:3] *= object_scale
    
    # calculate pose feasibility for each pose grid
    pose_map = check_pose(
                target_object_poses,
                target_object_sq_param, 
                other_objects_poses,
                temp_other_objects_sq_params, 
                depth_renderer)

    return pose_map
