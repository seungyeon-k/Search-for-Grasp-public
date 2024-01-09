import torch
import numpy as np
import matplotlib.pyplot as plt

from policy.pose_map_estimator import get_pose_map
from policy.reachability_map_estimator import get_graspability_hindrance, check_graspability, get_graspability_hindrance_batchwise
from policy.pick_and_place_planner import get_random_graspable_object_and_pose, place_object_at_random_position
from policy.push_planner import get_random_pushing_action, estimated_pushing_dynamics
from copy import deepcopy
from .utils import get_pc_of_objects, get_shelf_sq_values
from control.utils import render_map
from functions.LieGroup_torch import invSE3

import time
import open3d as o3d
from .utils import get_poses_from_grid

from models import load_pretrained

class SearchPolicy():
	def __init__(
		self,
		target_object_sq_param,
		depth_renderer,
		pose_map_grid,
		reachability_map_grid,
		reachability_grid_shape,
		gripper_open_pc,
		shelf_info,
		Ts_shelf,
		parameters_shelf,
		mode,
		device,
		step_num=3,
		sample_num=30,
		hinderance_score_weight=0.1,
		hinderance_mode='smooth',
		action_pool=["pick_and_place", "push"]
	):
		
		# initialize
		self.target_object_sq_param = target_object_sq_param.to(device)
		self.depth_renderer = depth_renderer
		self.pose_map_grid = pose_map_grid.to(device)
		self.reachability_map_grid = reachability_map_grid.to(device)
		self.reachability_grid_shape = reachability_grid_shape
		self.gripper_open_pc = gripper_open_pc.to(device)
		self.shelf_info = shelf_info
		self.device = device
		self.Ts_shelf = Ts_shelf
		self.parameters_shelf = parameters_shelf
		self.hinderance_mode = hinderance_mode
		self.discount_factor = 0.9
		self.hinderance_score_weight = hinderance_score_weight
		self.step_num = step_num
		self.sample_num = sample_num
		self.mode = mode # search_and_grasp, search_for_grasp
		self.action_pool = action_pool
		self.last_moved_obj_idx = []
		self.found_target = False
		
		# setup pushing dynamics model
		model_root = 'pretrained/sqpdnet_config'
		model_identifier = 'pretrained'
		model_config_file = 'sqpdnet_config.yml'
		model_ckpt_file = 'model_best.pkl'
		self.model, _ = load_pretrained(
			identifier=model_identifier,
			config_file=model_config_file,
			ckpt_file=model_ckpt_file,
			root=model_root
		)
		self.model = self.model.to(self.device)

	def target_detected(self, pose, target_idx):
		"""if target object is found, the policy is run with "found_target" mode.

		Args:
			pose (4 x 4 tensor): SE(3) pose matrix of target object
		"""
		self.found_target = True
		self.target_pose = pose
		self.target_idx = target_idx

	def update_pose_map(
		self,
		objects_poses,
		objects_sq_params,
	):
		"""update pose-existability map with given observed objects. 

		Args:
			other_objects_poses (n x 4 x 4 tensor): SE(3) pose matrices of observed non-target objects
			other_objects_sq_params (n x 5 tensor): superquadric parameters of observed non-target objects
		"""
		if not self.found_target:

			self.pose_map = get_pose_map(
				self.pose_map_grid,
				self.target_object_sq_param,
				objects_poses,
				objects_sq_params,
				self.depth_renderer
			)

			# render_map(
			# 	objects_poses,
			# 	objects_sq_params,
			# 	self.pose_map_grid,
			# 	self.pose_map,
			# 	self.shelf_info
			# )

			self.pose_map_grid = self.pose_map_grid[self.pose_map]
			
		else:
    			
			# if target object was found, confine the existable position to detected target position
			self.pose_map = torch.tensor([True])
			self.pose_map_grid = self.target_pose[0:2, 3].unsqueeze(0)
		
		return self.pose_map.sum() > 0

	def sample_action(
		self,
		objects_poses,
		objects_sq_params,
		pc_of_objects,
		remove_idxs
		):
		if "pick_and_place" in self.action_pool:
			graspable_object_idxs, grasping_poses = get_random_graspable_object_and_pose(
				objects_poses,
				objects_sq_params,
				pc_of_objects,
				self.gripper_open_pc,
				self.shelf_info,
				remove_idxs
			)
		else:
			graspable_object_idxs = []
  
		if "push" in self.action_pool:
			pushing_object_idxs, pushing_poses, pushing_directions, pushing_distances = get_random_pushing_action(
				objects_poses,
				objects_sq_params,
				pc_of_objects,
				self.gripper_open_pc,
				remove_idxs,
				self.Ts_shelf,
				self.parameters_shelf,
				pushing_directions = torch.tensor([[0., -1., 0.], [0., 1., 0.]]),
				pushing_distance = torch.tensor([[0.05, 0.10, 0.15]])
			)
		else:
			pushing_object_idxs = []

		grasping_object_num = len(graspable_object_idxs)
		pushing_object_num = len(pushing_object_idxs)

		if grasping_object_num == 0 and pushing_object_num == 0:
			return {
				"action_type" : "end",
			}

		random = torch.randint(grasping_object_num + pushing_object_num, size=[])

		if random < grasping_object_num:
			sampled_object = graspable_object_idxs[random]
			grasping_pose = grasping_poses[random][torch.randint(len(grasping_poses[random]), size=[])]
			return {
				"action_type" : "grasping",
				"sampled_object" : sampled_object,
				"grasping_pose" : grasping_pose
			}
		else:
			random -= grasping_object_num
			sampled_object = pushing_object_idxs[random]
			push_pose = pushing_poses[random]
			pushing_direction = pushing_directions[random]
			pushing_distance = pushing_distances[random]
			return {
				"action_type" : "pushing",
				"sampled_object" : sampled_object,
				"pushing_pose" : push_pose,
				"pushing_direction" : pushing_direction,
				"pushing_distance" : pushing_distance
			}
	
	def evaluate(
		self,
		objects_poses,
		objects_sq_params,
		pose_map_grid,
	):
		with torch.no_grad():
			if not self.found_target:

				# tic = time.time()
				predicted_pose_map = get_pose_map(
					pose_map_grid,
					self.target_object_sq_param,
					objects_poses,
					objects_sq_params,
					self.depth_renderer
				)
				# toc = time.time()
				# print(f"elasped time for pose map calculation : {toc-tic}")

				# convert grid map to SE(3)s
				SE3s = get_poses_from_grid(pose_map_grid, height_offset=self.target_object_sq_param[2])
			else:
				predicted_pose_map = torch.tensor(True)
				SE3s = self.target_pose.unsqueeze(0)

			if self.mode == "search_for_grasp" or (self.mode == "search_and_grasp" and self.found_target):
				hindrance_score_list = []
				pc_of_target_objects = get_pc_of_objects(SE3s, self.target_object_sq_param.repeat(len(SE3s), 1))
				tic = time.time()
				# for SE3, pc_of_target_object in zip(SE3s, pc_of_target_objects):
				# 	hindrance_score = get_graspability_hindrance(
				# 		SE3,
				# 		self.target_object_sq_param,
				# 		objects_poses,
				# 		objects_sq_params,
				# 		pc_of_target_object,
				# 		self.shelf_info,
				# 		self.gripper_open_pc,
				# 		visualize=False,
      			# 		mode=self.hinderance_mode
				# 	)
				# 	hindrance_score_list.append(hindrance_score)
				# hindrance_score_list = torch.tensor(hindrance_score_list)
				hindrance_score = get_graspability_hindrance_batchwise(
					SE3s,
					self.target_object_sq_param,
					objects_poses,
					objects_sq_params,
					pc_of_target_objects,
					self.shelf_info,
					self.gripper_open_pc,
					visualize=False,
					mode=self.hinderance_mode
				)
				toc = time.time()
				print(f"elasped time for grapability map calculation : {toc-tic}")

			if self.mode == "search_for_grasp":
				score = -torch.sum(hindrance_score_list) * self.hinderance_score_weight - torch.sum(predicted_pose_map)
			elif self.mode == "search_and_grasp" and not self.found_target :
				score = -torch.sum(predicted_pose_map).float()
			elif self.mode == "search_and_grasp" and self.found_target :
				score = -torch.sum(hindrance_score_list).float() * self.hinderance_score_weight
			return score, pose_map_grid[predicted_pose_map]

	def sample_trajectories(
		self,
		objects_poses,
		objects_sq_params
	):
		"""sample trajectories

		Args:
			other_objects_poses (_type_): _description_
			other_objects_sq_params (_type_): _description_

		Returns:
			_type_: _description_
		"""
		initial_score, pose_map_grid = self.evaluate(objects_poses, objects_sq_params, self.pose_map_grid)
		# print(f"init score is {initial_score}")
		total_score_list = []
		saved_action = []
		success_sampling = False
		for sample_idx in range(self.sample_num):
      
			current_object_poses = deepcopy(objects_poses)
			current_score = deepcopy(initial_score)
			total_score = deepcopy(initial_score)
			temp_pose_map_grid = deepcopy(pose_map_grid)
			traj_score_list = []
			success_traj = False
   
			if not self.found_target:
				remove_idxs = []
			else:
				remove_idxs = [self.target_idx]
    
			for step in range(self.step_num):
				pc_of_objects = get_pc_of_objects(current_object_poses, objects_sq_params)
				# sample action
				action = self.sample_action(objects_poses, objects_sq_params, pc_of_objects, remove_idxs)
				# predict dynamics
				if step == 0:
					saved_action.append(action)
				if action["action_type"] == 'grasping':
					placing_grid = deepcopy(self.reachability_map_grid)
					placing_grid[:,2] = current_object_poses[action["sampled_object"], 2, 3]
					# add target object on every possible pose for collision checking
					if not self.found_target:
						SE3s = get_poses_from_grid(temp_pose_map_grid, self.target_object_sq_param[2])
						temp_objects_poses = torch.cat([current_object_poses, SE3s], dim=0)
						temp_objects_sq_params = torch.cat([objects_sq_params, self.target_object_sq_param.repeat(len(SE3s), 1)], dim=0)
					else:
						temp_objects_poses = deepcopy(current_object_poses)
						temp_objects_sq_params = deepcopy(objects_sq_params)
					placed_pose = place_object_at_random_position(	
						temp_objects_poses,
						temp_objects_sq_params,
						pc_of_objects,
						self.gripper_open_pc,
						action["sampled_object"],
						action["grasping_pose"],
						placing_grid,
						self.reachability_grid_shape,
						self.shelf_info
					)
					if placed_pose is None: # impossible to place the object
						break
					elif step == 0:
						current_object_poses[action["sampled_object"]] = placed_pose
						saved_action[-1]["placed_object_pose"] = current_object_poses[action["sampled_object"]]
				elif action["action_type"] == 'pushing':
					current_object_poses = estimated_pushing_dynamics(
						current_object_poses,
						objects_sq_params,
						action["sampled_object"],
						action["pushing_pose"][0:3,3],
						action["pushing_direction"][0:2],
						self.model
					)
				elif action["action_type"] == 'end':
					break

				# evaluate
				score, temp_pose_map_grid = self.evaluate(current_object_poses, objects_sq_params, temp_pose_map_grid)
				# print(f"{sample_idx}-{step} : {score}")
				score_diff = score - current_score
				total_score += score_diff * (self.discount_factor ** step)
				traj_score_list.append(total_score)
				current_score = score
				remove_idxs.append(int(action["sampled_object"]))
				success_sampling = True
				success_traj = True
				if len(temp_pose_map_grid) == 0:
					break
			if success_traj:
				traj_score = torch.stack(traj_score_list).max()
				total_score_list.append(traj_score)
			else:
				total_score_list.append(torch.tensor(-9999.0))
		
		if success_sampling:
			sorted_idx = torch.tensor(total_score_list).argsort(descending=True)

			return [saved_action[idx] for idx in sorted_idx]
		else:
			return []

	def find_best_action(
		self,
		objects_poses,
		objects_sq_params,
	):
		"""_summary_

		Args:
			other_objects_poses (_type_): _description_
			other_objects_sq_params (_type_): _description_

		Returns:
			_type_: _description_
		"""
		
		tic = time.time()

		pose_map_update_success = self.update_pose_map(
			objects_poses,
			objects_sq_params
		)

		if not pose_map_update_success:
			return[{
				"action_type" : "fail",
				"description" : "pybullet image and depth render prediction for target does not match"
			}]
  
		actions = self.sample_trajectories(objects_poses, objects_sq_params)

		toc = time.time()
		print(f"elasped time for search policy : {toc-tic}")

		action_infos = []

		for action in actions:
			if action["action_type"] == 'grasping':
				if "placed_object_pose" in action:
					
					object_id = action["sampled_object"]
					initial_object_pose = objects_poses[object_id]
					final_object_pose = action["placed_object_pose"]
					initial_grasp_pose = action["grasping_pose"]
					final_grasp_pose = final_object_pose @ invSE3(initial_object_pose.unsqueeze(0)).squeeze() @ initial_grasp_pose
	
					action_info = {
						"action_type" : "grasping",
						"grasping_object_id" : object_id,
						"initial_object_pose" : initial_object_pose,
						"final_object_pose" : final_object_pose,
						"initial_grasp_pose" : initial_grasp_pose,
						"final_grasp_pose" : final_grasp_pose,
					}
     
					action_infos.append(action_info)

			elif action["action_type"] == 'pushing':
				object_id = action["sampled_object"]
				initial_push_pose = action["pushing_pose"]
				final_push_pose = deepcopy(initial_push_pose).to(self.device)
				final_push_pose[0:3,3] += action["pushing_distance"].to(self.device) * action["pushing_direction"].to(self.device)
				action_info = {
					"action_type" : "pushing",
					"pushing_object_id" : object_id,
					"pushing_direction" : action["pushing_direction"],
					"pushing_distance" : action["pushing_distance"],
					"initial_push_pose" : initial_push_pose,
					"final_push_pose" : final_push_pose
				}

				action_infos.append(action_info)
    
		if len(action_infos) == 0:
			return [{
				"action_type" : "fail",
				"description" : "failed to sample feasible action"
			}]
		else:
			return action_infos
	