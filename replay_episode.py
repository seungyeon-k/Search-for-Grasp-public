import torch
import os
import numpy as np
from control.control_env import ControlSimulationEnv
import pickle
import pybullet as p
from glob import glob
from policy.reachability_map_estimator import (
 	check_graspability
)
from policy.utils import get_pc_of_objects
from control.gripper import Gripper

#######################################################################
############################## directory ##############################

idx = 41
#exp_root_folder_name = f'exp_results_sim/box_object_4/R_search_for_grasp_box/{idx}'
exp_root_folder_name = f'../20230808_data/target_box/objects_8/R_search_for_grasp/{idx}'

#######################################################################
#######################################################################

# make video
log_paths = glob(os.path.join(exp_root_folder_name, 'data_iter_*.pkl'))
len_episode = len(log_paths)
logs = []
for i in range(len_episode):
    with open(os.path.join(exp_root_folder_name, f'data_iter_{i}.pkl'), 'rb') as f:
        data = pickle.load(f)
        logs.append(data)
initial_scenario_path = os.path.join(exp_root_folder_name, 'initial_scenario.pkl')

env = ControlSimulationEnv(enable_gui=True)
p.resetDebugVisualizerCamera(0.01, -115.7616444, 0.4165967, [-0.04457319, 0.44833383, 0.65361773])
gripper_open = Gripper(np.eye(4), 0.08)
gripper_open_pc = gripper_open.get_gripper_afterimage_pc(
    pc_dtype='torch'
)

target_sq = torch.tensor([0.02, 0.02, 0.05, 0.2, 1.0])
action_keys = ['initial_grasp_pose', 'final_grasp_pose', 'initial_push_pose', 'final_push_pose', 'grasp_pose']
final_log = logs[-1]

target_parameter = torch.from_numpy(final_log['target_parameter'])
try:
    gt_surrounding_objects_poses = torch.from_numpy(final_log['gt_surrounding_objects_poses'])
    gt_surrounding_objects_parameters = torch.from_numpy(final_log['gt_surrounding_objects_parameters'])
except:
    gt_surrounding_objects_poses = torch.from_numpy(final_log['surrounding_objects_poses'])
    gt_surrounding_objects_parameters = torch.from_numpy(final_log['surrounding_objects_parameters'])
pc_of_target_object = get_pc_of_objects(gt_surrounding_objects_poses[-1].unsqueeze(0), target_parameter.unsqueeze(0)).squeeze()

if final_log['status'] == 'success':
    graspability, grasp_pose = check_graspability(
        gt_surrounding_objects_poses[-1],
        target_parameter,
        gt_surrounding_objects_poses[:-1],
        gt_surrounding_objects_parameters[:-1],
        pc_of_target_object,
        env.sim._get_shelf_info(),
        gripper_open_pc,
        visualize=False
        )
    print(f"graspable : {graspability}, num of graspable pose : {len(grasp_pose)}")
else:
    print("failed scenario")
graspability, grasp_pose = check_graspability(
        gt_surrounding_objects_poses[-1],
        target_parameter,
        gt_surrounding_objects_poses[[]],
        gt_surrounding_objects_parameters[[]],
        pc_of_target_object,
        env.sim._get_shelf_info(),
        gripper_open_pc,
        visualize=False
    )
env.load_objects(initial_scenario_path)
for log in logs:

    if log['status'] == 'success':
        if graspability:
            action = {}
            action['action_type'] = 'target_retrieval'
            action['grasp_pose'] = grasp_pose[-1]
            env.implement_action(action)
    else:
        action = log['implemented_action']
        for key in action:
            if key in action_keys:
                action[key] = torch.from_numpy(action[key])
        print(action)
        print(log['status'])
        
        target_parameter = torch.from_numpy(log['target_parameter'])
        # try:
        gt_surrounding_objects_poses = torch.from_numpy(log['gt_surrounding_objects_poses'])
        gt_surrounding_objects_parameters = torch.from_numpy(log['gt_surrounding_objects_parameters'])
        # except:
        #     gt_surrounding_objects_poses = torch.from_numpy(log['surrounding_objects_poses'])
        #     gt_surrounding_objects_parameters = torch.from_numpy(log['surrounding_objects_parameters'])
        print(gt_surrounding_objects_poses[-1])
        pc_of_target_object = get_pc_of_objects(gt_surrounding_objects_poses[-1].unsqueeze(0), target_parameter.unsqueeze(0)).squeeze()
        graspability, grasp_pose = check_graspability(
            gt_surrounding_objects_poses[-1],
            target_parameter,
            gt_surrounding_objects_poses[:-1],
            gt_surrounding_objects_parameters[:-1],
            pc_of_target_object,
            env.sim._get_shelf_info(),
            gripper_open_pc,
            visualize=False
            )
        
        env.implement_action(action)


# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, f'video/video_{idx}.mp4')
# env.load_objects(initial_scenario_path)
# for log in logs:
#     if log['status'] == 'success':
#         if graspability:
#             action = {}
#             action['action_type'] = 'target_retrieval'
#             action['grasp_pose'] = grasp_pose[-1]
#             env.implement_action(action)
#     else:
#         action = log['implemented_action']
#         for key in action:
#             if key in action_keys:
#                 action[key] = torch.from_numpy(action[key])
#         print(action)
#         print(log['status'])
#         env.implement_action(action)

# p.stopStateLogging()