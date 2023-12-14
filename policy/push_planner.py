import torch
from .reachability_map_estimator import check_collision
from copy import deepcopy
from functions.lie_torch import vectorize_scene_data_torch, get_SE3s_torch, quats_to_matrices_torch
import numpy as np


tilt = torch.tensor(torch.pi/6)
gripper_orientation = torch.tensor([[torch.cos(tilt), 0., torch.sin(tilt)],[0., 1., 0.],[-torch.sin(tilt), 0., torch.cos(tilt)]]) @ torch.tensor([[[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]], [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]])

def push_planner(
    object_pcs,
    gripper_pc,
    object_poses,
    gripper_orientation=gripper_orientation,
    pushing_directions=torch.tensor([[0., -1., 0.], [0., 1., 0.]]),
    gripper_depth=-0.085,
    gripper_height=0.01,
    spacing_width=0.0,
	data_type='torch'
):
    """_summary_

    Args:
        object_poses (n_obj x 4 x 4 tensor): poses of objects represented in global frame
        object_pcs (n_obj x n_pc x 3 tensor): pointclouds of objects represented in global frame
        gripper_pc (n_gripper_pc x 3 tensor): _description_
        gripper_orientation (3 x 3 tensor) or (n_direction x 3 x 3 tensor): _description_. Defaults to torch.tensor([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]).
        pushing_directions (n_direction x 3 tensor, optional): _description_. Defaults to torch.tensor([[0., -1., 0.], [0., 1., 0.]]).
        gripper_depth (float, optional): _description_. Defaults to 0.03.
        spacing_width (float, optional): _description_. Defaults to 0.03.

    Returns:
        _type_: _description_
    """
    
    if data_type == 'torch':
        if len(gripper_orientation.shape) == 2:
            gripper_orientation = gripper_orientation.repeat(len(pushing_directions), 1, 1)
            
        gripper_orientation = gripper_orientation.to(object_pcs.device)
        pushing_directions = pushing_directions.to(object_pcs.device)
        
        rotated_gripper_pc = (gripper_pc.to(object_pcs.device) @ gripper_orientation.permute(1, 2, 0)).permute(1, 2, 0) # n_gripper_pc x n_direction x 3
        object_pushing_point =((object_pcs - object_poses[:,0:3,3].unsqueeze(1)) @ pushing_directions.T).min(dim=1).values # n_obj x n_direction
        gripper_pushing_point = (rotated_gripper_pc * pushing_directions).sum(dim=-1).max(dim=0).values # n_direction
        trans = (object_pushing_point - gripper_pushing_point.unsqueeze(0) - spacing_width).unsqueeze(-1) * pushing_directions # n_obj x n_direction x 3
        
        pushing_pose = torch.eye(4).repeat(len(object_pcs), len(pushing_directions), 1, 1).to(gripper_pc.device)
        pushing_pose[..., 0:3, 0:3] = gripper_orientation
        pushing_pose[..., 0:3, 3] = trans + object_poses[:,0:3,3].unsqueeze(1) + (gripper_orientation @ torch.tensor([0, 0, gripper_depth]).to(gripper_pc.device)).unsqueeze(0)

        object_floor_height = object_pcs[:,:,2].min(dim=-1).values # n_obj
        gripper_floor_height = rotated_gripper_pc[:, :, 2].min(dim=0).values # n_direction

        pushing_pose[..., 2, 3] = object_floor_height.unsqueeze(-1) - gripper_floor_height.unsqueeze(0) + gripper_height
    
    elif data_type == 'numpy':
        if len(gripper_orientation.shape) == 2:
            gripper_orientation = np.repeat(np.expand_dims(gripper_orientation, axis=0), len(pushing_directions), axis=0)
        
        rotated_gripper_pc = (gripper_pc @ gripper_orientation.transpose((1, 2, 0))).transpose((1, 2, 0))
        object_pushing_point =(
            (
                object_pcs - np.expand_dims(object_poses[:, 0:3, 3], 1)
            ) @ pushing_directions.T
        ).min(axis=1) # n_obj x n_direction
        gripper_pushing_point = (
            rotated_gripper_pc * pushing_directions
        ).sum(axis=-1).max(axis=0) # n_direction
        trans = np.expand_dims(
            object_pushing_point - np.expand_dims(gripper_pushing_point, 0) - spacing_width,
            -1
        ) * pushing_directions # n_obj x n_direction x 3
        
        pushing_pose = np.expand_dims(np.eye(4), [0, 1]).repeat(len(object_pcs), 0).repeat(len(pushing_directions), 1)
        pushing_pose[..., 0:3, 0:3] = gripper_orientation
        pushing_pose[..., 0:3, 3] = (
            trans 
            + np.expand_dims(object_poses[:,0:3,3], 1) 
            + np.expand_dims(gripper_orientation @ np.array([0, 0, gripper_depth]), 0)
        )
        object_floor_height = object_pcs[:,:,2].min(axis=-1) # n_obj
        gripper_floor_height = rotated_gripper_pc[:, :,2].min(axis=0)

        pushing_pose[..., 2, 3] = np.expand_dims(object_floor_height, 1).repeat(len(pushing_directions), 1) - np.expand_dims(gripper_floor_height, 0).repeat(len(object_poses), 0) + gripper_height
    
    return pushing_pose # n_obj x n_direction x 4 x 4    

def get_random_pushing_action(
    objects_poses,
    objects_sq_params,
    pc_of_objects,
    pc_of_gripper,
    remove_idxs=[],
    Ts_shelf=None,
    parameters_shelf=None,
    pushing_directions=torch.tensor([[0., -1., 0.], [0., 1., 0.]]),
    pushing_distance=torch.tensor([[0.03, 0.06, 0.09]])
):
    """randomly select an object and pushable pose from observed non-target objects. (on the shelf)
       note that sampling is restricted to graspable objects and valid pushing pose.

    Args:
        objects_poses (n_obj x 4 x 4 tensor): poses of objects represented in global frame
        objects_sq_params (n_obj x 5 tensor): superquadric parameters of objects
        pc_of_objects (n_obj x n_pc_obj x 3 tensor): pointclouds of objects represented in global frame
        pc_of_gripper (n_pc_gripper x 3 tensor): pointcloud of gripper represented in gripper body frame
        remove_idxs (list of intergers, optional): indices of object no to be selected. Defaults to [].
        Ts_shelf (n_comp x 4 x 4 tensor, optional): poses of shelf components represented in global frame. Defaults to None.
        parameters_shelf (n_comp x 5 tensor, optional): superquadric parameters of shelf components. Defaults to None.
        pushing_directions (n_dir x 3 tensor, optional): dircetion of pushing, which should be normal vector. Defaults to torch.tensor([[0., -1., 0.], [0., 1., 0.]]).
        pushing_distance (n_dist tensor, optional): distance to push. Defaults to torch.tensor([[0.03, 0.06, 0.09]]).

    Returns:
        available_obj_indices
        available_push_poses
        available_directions
        random_pushing_distance
    """
    
    push_poses = push_planner(pc_of_objects, pc_of_gripper, objects_poses, pushing_directions=pushing_directions)
    
    available = ~check_collision(
        objects_poses, 
        objects_sq_params,
        push_poses.reshape(-1, 4, 4), 
        pc_of_gripper, 
        Ts_shelf,
        parameters_shelf,
        visualize=False
    )
    available = available.reshape(-1, 2)
    available[[remove_idxs],:] = False
    
    available_push_poses = push_poses[available]
    available_obj_indices = (torch.arange(0, len(objects_poses)).repeat(2,1).T).to(objects_poses.device)[available]
    available_directions = pushing_directions.repeat(len(objects_poses), 1, 1).to(objects_poses.device)[available]
    random_pushing_distance = pushing_distance[torch.randint(len(pushing_distance), size=[len(available_directions)])]

    return available_obj_indices, available_push_poses, available_directions, random_pushing_distance

def estimated_pushing_dynamics(
    objects_poses,
    objects_sq_params,
    object_to_be_pushed,
    pushing_position,
    pushing_direction,
    model
):
    """randomly select an object and pushable pose from observed non-target objects. 
    note that sampling is restricted to graspable objects and valid pushing pose.

    Args:
        objects_poses (n x 4 x 4 tensor): SE(3) pose matrices of observed non-target objects
        objects_sq_params (n x 5 tensor): superquadric parameters of observed non-target objects
        object_to_be_pushed (int): index of object to be pushed
        pushing_position (3 tensor): initial position of the pushing action
        pushing_direction (3 tensor): direction of the pushing action

    Returns:
        objects_poses_updated (n x 4 x 4 tensor) : poses of objects after pushing
    """

    # num objects
    num_objects = 4
    num_primitives = 5

    # updated pose initialize
    objects_poses_updated = deepcopy(objects_poses)

    # set closest 4 objects
    pushed_object_position = objects_poses[object_to_be_pushed, :3, 3]
    closest_object_indices = torch.argsort(
        torch.norm(
            objects_poses[:, :3, 3] - pushed_object_position.unsqueeze(0),
            dim=1
        ),
    )
    Ts_old = objects_poses[closest_object_indices[:num_objects]]
    parameters = objects_sq_params[closest_object_indices[:num_objects]]

    # vectorize inputs
    if len(Ts_old) < num_objects: 
        num_objects = len(Ts_old)
    x_scene = torch.zeros(num_objects, 13, num_primitives).to(objects_poses.device)
    a_scene = torch.zeros(num_objects, 5).to(objects_poses.device)
    for object_id in range(num_objects):

        # change object indices
        object_idxs = list(range(num_objects))
        object_idxs.remove(object_id)
        Ts_obj_centric = Ts_old[[object_id] + object_idxs]
        parameters_obj_centric = parameters[
            [object_id] + object_idxs
        ]

        # vectorize data
        x, a = vectorize_scene_data_torch(
            Ts_obj_centric, 
            parameters_obj_centric, 
            pushing_position, 
            pushing_direction, 
            num_primitives=num_primitives,
            motion_dim='3D'
        )

        # update data
        x_scene[object_id] = x
        a_scene[object_id] = a

    # pushing dynamics forward
    motion_preds = model(x_scene, a_scene)
    motion_preds = motion_preds.detach()

    # calculate predicted pose
    Ts_diff_pred = get_SE3s_torch(
        quats_to_matrices_torch(motion_preds[:, 3:7]), 
        motion_preds[:, :3]
    ).to(objects_poses.device)
    Ts_new_pred = Ts_old @ Ts_diff_pred

    # update pose
    objects_poses_updated[closest_object_indices[:num_objects]] = Ts_new_pred

    return objects_poses_updated