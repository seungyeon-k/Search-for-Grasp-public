import os
from re import T
import numpy as np
import torch
from copy import deepcopy
import pybullet as p
from datetime import datetime
import time
import matplotlib.pyplot as plt
import pickle
import open3d as o3d
import torch.nn.functional as F
from functions.lie_torch import quats_to_matrices_torch, get_SE3s_torch, matrices_to_quats_torch, exp_so3
from functions.lie import matrices_to_quats
from control.control_env import ControlSimulationEnv
from control.utils import (
	render_segmentation,
	render_objects,
	render_objects_with_pc,
	render_map,
	render_graspability,
	get_shelf_bounds,
	get_shelf_grid,
 	get_shelf_sq_values,
	add_shelf,
	add_surrounding_objects,
)

# policy 
from policy.depth_render import DepthRenderer
from policy.pose_map_estimator import (
	get_pose_map,
	check_depth_feasibility
)
from policy.reachability_map_estimator import (
 	check_graspability
)
from control.gripper import Gripper
from policy.search_policy import SearchPolicy
from policy.utils import get_pc_of_objects
from functions.communicator_server import Listener

# models
from models.segmentation_model_loader import (
	SegmentationModel,
	compute_xyz
)
from models import load_pretrained
from loaders.segmentation_dataset import normalize_pointcloud as normalize_pointcloud_seg
from loaders.recognition_dataset import normalize_pointcloud as normalize_pointcloud_recog
from loss.segmentation_loss import hungarian_matching, batch_reordering

class Controller:
	def __init__(
			self, 
			cfg_controller, 
			device='cpu',
			enable_gui=False
		):

		# get inputs
		self.device = device
		self.realworld = cfg_controller.realworld
		self.str_realworld = 'real' if self.realworld else 'sim'
		self.max_iter = cfg_controller.max_iter
		self.mode = cfg_controller.mode # 'search_for_grasp', 'search_and_grasp'
		self.recognition = cfg_controller.recognition
		self.str_recognition = 'R' if self.recognition else 'O' 
		self.cfg_controller = cfg_controller
		self.save_dir = cfg_controller.get('save_dir', datetime.now().strftime('%Y%m%d-%H%M'))
		self.load_states_dir = cfg_controller.get('load_states_dir', None)
		self.hinderance_score_weight = cfg_controller.get('hinderance_score_weight', 0.1)
		self.target_object = cfg_controller.objects.get('target', 'cylinder')
		if not self.realworld:
			self.num_objects = cfg_controller.objects.num_objects
		if self.realworld:
			self.target_sq = torch.tensor([0.02, 0.02, 0.05, 0.2, 1.0]).to(self.device)
		self.grid_size = cfg_controller.get('grid_size', 'large')
		self.grid_dim = cfg_controller.get('grid_dim', '2D')
		self.action_pool = cfg_controller.get('action_pool', ["pick_and_place", "push"])

		# debugging mode
		self.realworld_debug = False

		# setup environment
		if self.realworld:
			self.env = ControlSimulationEnv(enable_gui=False)
			camera_param = self.env.sim.camera_params[0]
			self.depth_renderer = DepthRenderer(camera_param, device=device)
		else:
			self.env = ControlSimulationEnv(enable_gui=enable_gui)
			camera_param = self.env.sim.camera_params[0]
			self.depth_renderer = DepthRenderer(camera_param, device=device)
			p.resetDebugVisualizerCamera(0.01, -115.7616444, 0.4165967, [-0.04457319, 0.44833383, 0.65361773])
		self.shelf_info = self.env.sim._get_shelf_info()

		# setup segmentation / recognition model
		self.calibration_k = 10
		self.num_pts_recog = 100
		self.segmentation_model = self.load_segmentation_network()
		self.recognition_model = self.load_recognition_network()
		self.segmentation_model.eval()
		self.recognition_model.eval()

		# communication setting
		if self.realworld:
			self.ip = cfg_controller.ip
			self.port = cfg_controller.port
			# self.table_offest = 0.002

		# load gripper
		gripper_open = Gripper(np.eye(4), 0.08)
		self.gripper_open_pc = gripper_open.get_gripper_afterimage_pc(
			pc_dtype='torch'
		).to(self.device)

		# set workspace
		self.shelf_bounds_min, self.shelf_bounds_max = get_shelf_bounds(
			self.shelf_info
		)
		self.workspace_bounds = torch.tensor(self.env.workspace_bounds).float()
		self.Ts_shelf, self.parameters_shelf = get_shelf_sq_values(self.shelf_info, device=self.device)

		# make grids
		self.pose_map_grid, _ = get_shelf_grid(
			self.shelf_info, resolution=0.015, dtype='torch', grid_size=self.grid_size, grid_dim=self.grid_dim
		) # n_grid x 3 or n_grid x 4,  resolution=0.01
		self.reachability_grid, self.reachability_grid_shape = get_shelf_grid(
			self.shelf_info, resolution=0.015, dtype='torch', grid_size=self.grid_size
		) # n_grid x 3 resolution=0.01
		
		# make map grids
		self.pose_map_grid = self.pose_map_grid.to(self.device)
		self.reachability_grid = self.reachability_grid.to(self.device)

	def reset_env(self, states_dir=None):
		self.env.sim.reset_robot()
		self.env.sim.robot_go_home()
		if states_dir is None:
			self.env.reset(self.cfg_controller.objects)
			# set target objects
			if self.target_object == 'cylinder':
				self.target_sq = torch.tensor([0.02, 0.02, 0.05, 0.2, 1.0]).to(self.device)
			if self.target_object == 'box':
				self.target_sq = torch.tensor([0.015, 0.02, 0.05, 0.2, 0.2]).to(self.device)

			spawn_success = self.spawn_target_object()
	
			# initialize policy searcher
			self.search_policy = SearchPolicy(
				self.target_sq, 
				self.depth_renderer, 
				self.pose_map_grid,
				self.reachability_grid,
				self.reachability_grid_shape,
				self.gripper_open_pc, 
				self.shelf_info,
				self.Ts_shelf,
				self.parameters_shelf,
				mode=self.mode,
				device=self.device,
				hinderance_score_weight=self.hinderance_score_weight,
				action_pool = self.action_pool
			)
			return spawn_success
		else:
			if self.target_object == 'cylinder':
				self.target_sq = torch.tensor([0.02, 0.02, 0.05, 0.2, 1.0]).to(self.device)
			if self.target_object == 'box':
				self.target_sq = torch.tensor([0.015, 0.02, 0.05, 0.2, 0.2]).to(self.device)
			self.env.load_objects(states_dir)
   			# initialize policy searcher
			self.search_policy = SearchPolicy(
				self.target_sq, 
				self.depth_renderer, 
				self.pose_map_grid,
				self.reachability_grid,
				self.reachability_grid_shape,
				self.gripper_open_pc, 
				self.shelf_info,
				self.Ts_shelf,
				self.parameters_shelf,
				mode=self.mode,
				device=self.device,
				hinderance_score_weight=self.hinderance_score_weight,
				action_pool = self.action_pool
			)
			return True

	def get_search_policy_realworld(self):
		self.env.sim.reset_robot()
		self.env.sim.robot_go_home()

		self.search_policy = SearchPolicy(
			self.target_sq, 
			self.depth_renderer, 
			self.pose_map_grid,
			self.reachability_grid,
			self.reachability_grid_shape,
			self.gripper_open_pc, 
			self.shelf_info,
			self.Ts_shelf,
			self.parameters_shelf,
			mode=self.mode,
			device=self.device,
			hinderance_score_weight=self.hinderance_score_weight,
			action_pool = self.action_pool
		)
		return True

	def spawn_datas(self, exp_idx):
		# data save folder
		save_folder = os.path.join(
			'exp_results',
   			self.save_dir,
			'initial_scenarios'
		)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
		
		# spawn every objects
		spawn_success = self.reset_env()
		Ts, _ = self.env.groundtruth_recognition(
					output_dtype='torch'
				)
		is_in_workspace = self.check_objects_are_in_workspace(self.workspace_bounds, Ts.to(self.device))
  
		# save scenario
		if not spawn_success or not is_in_workspace:
			#print(f"spawn success : {spawn_success}, is_in_workspace : {is_in_workspace}")
			return False # end control!
		else:
			#print('success')
			with open(os.path.join(save_folder, f'initial_scenario_{exp_idx}.pkl'), 'wb') as f:
				pickle.dump(self.env.object_infos, f, pickle.HIGHEST_PROTOCOL)
 			
			return True

	#############################################################
	######################## CONTROLLER #########################
	#############################################################
 
	def control(self, exp_idx=0):
		
		# data save folder
		save_folder = os.path.join(
			f'exp_results_{self.str_realworld}', 
   			self.save_dir,
			f'{self.str_recognition}_{self.mode}_{self.target_object}',
			str(exp_idx)
		)
		if not os.path.exists(save_folder):
			os.makedirs(save_folder)
		
		# spawn every objects
		if not self.realworld:
			if self.load_states_dir is not None:
				states_dir = os.path.join(self.load_states_dir, f'initial_scenario_{exp_idx}.pkl')
			else:
				states_dir = None

			spawn_success = self.reset_env(states_dir)

			# save scenario
			if not spawn_success:
				return False # end control!
			else:
				with open(os.path.join(save_folder, 'initial_scenario.pkl'), 'wb') as f:
					pickle.dump(self.env.object_infos, f, pickle.HIGHEST_PROTOCOL)

		# initialize serach policy
		if self.realworld:
			self.get_search_policy_realworld()

		# initialize data
		iteration = 0
		graspability = torch.tensor(False)
		found = False

		# control
		while not graspability or not found:
			
			#print(f"************************* Iteration {iteration} *************************")
	
			file_name = os.path.join(
				save_folder, 
				f'data_iter_{iteration}.pkl'
			)		
	
			# get observation
			if self.realworld:
				# load
				if not self.realworld_debug:	
					server = Listener(self.ip, self.port)
					#print(f'waiting image ...')
					data = server.recv_vision()
					np.save('test', data)
				else:
					data = np.load('test.npy', allow_pickle=True).item()
				# process data
				pc = data[b'pc']
				if 'labels' in data:
					labels = data[b'labels']
				else:
					labels = None
				color_image = data[b'color_image']
				depth_image = data[b'depth_image']
				del data

			else:
				pc, labels, color_image, depth_image, mask = self.env.observe()  
				# render_segmentation(pc, labels)
				
			# observation numpy to torch
			pc = pc.T
			pc = torch.tensor(pc).float().unsqueeze(0).to(self.device)
			if labels is not None:
				labels = torch.tensor(labels).float().unsqueeze(0).to(self.device)

			# recognition
			if self.recognition:
				if self.realworld: # 
					Ts, parameters = self.realworld_recognition(
						pc,
						labels=labels
					)						
				else: # gt segmentation label in simulation
					Ts, parameters = self.simulation_recognition(
						pc,
						labels
					)
					Ts_gt, parameters_gt = self.env.groundtruth_recognition(
					output_dtype='torch'
					)
					Ts_gt = Ts_gt.to(self.device)
					parameters_gt = parameters_gt.to(self.device)
			else:	
				Ts, parameters = self.env.groundtruth_recognition(
					output_dtype='torch'
				)
				# to device
				Ts = Ts.to(self.device)
				parameters = parameters.to(self.device)
	
				Ts_gt, parameters_gt = self.env.groundtruth_recognition(
				output_dtype='torch'
				)
				Ts_gt = Ts_gt.to(self.device)
				parameters_gt = parameters_gt.to(self.device)

			# save data dict initialize
			save_dict = {
				# 'color_image': color_image,
				# 'depth_image': depth_image,
				'pc': pc.squeeze(0).cpu().numpy().T,
				# 'label': labels,
				'surrounding_objects_poses': Ts.cpu().numpy(),
				'surrounding_objects_parameters': parameters.cpu().numpy(),
				'target_parameter': self.target_sq.cpu().numpy(),
			}
			data_to_send = {
				'list_sq_poses': Ts.cpu().numpy(),
				'list_sq_parameters': parameters.cpu().numpy(),				
			}

			if not self.realworld and self.recognition:
				save_dict['gt_surrounding_objects_poses'] = Ts_gt.cpu().numpy()
				save_dict['gt_surrounding_objects_parameters'] = parameters_gt.cpu().numpy()
	
			# check if target object is found
			if not found:
				if self.realworld:
					found = False
					# raise NotImplementedError # output found (detection module)
				else:
					found = (mask == self.num_objects + 4).sum() > 100

			# obtain task status
			if found: # found

				if self.realworld:
					raise NotImplementedError # output target pose	
				else:
					target_pose = self.env.groundtruth_recognition(output_dtype='torch')[0][-1].to(self.device)
	
				if self.recognition:
					recognized_target = (labels == self.num_objects + 4).sum() > self.num_pts_recog
	 
				pc_of_target_object = get_pc_of_objects(target_pose.unsqueeze(0), self.target_sq.unsqueeze(0)).squeeze().to(self.device)
	
				if self.realworld:
					raise NotImplementedError
				elif self.recognition:
					if recognized_target:
						target_idx = len(Ts) - 1
						Ts[-1] = target_pose
						parameters[-1] = self.target_sq
					else:
						target_idx = len(Ts)
						Ts = torch.cat([Ts, target_pose.unsqueeze(0)], dim=0)
						parameters = torch.cat([parameters, self.target_sq.unsqueeze(0)], dim=0)
				else:
					target_idx = len(Ts) - 1
	 
				graspability, grasp_pose = check_graspability(
					target_pose,
					self.target_sq,
					Ts[torch.arange(len(Ts)) != target_idx],
					parameters[torch.arange(len(parameters)) != target_idx],
					pc_of_target_object,
					self.shelf_info,
					self.gripper_open_pc,
					visualize=False
					)
    
				self.search_policy.target_detected(target_pose, target_idx)
				
				if not torch.is_tensor(graspability):
					graspability = torch.tensor(graspability)
				if graspability: # found and graspable
					#print("try target retrieve!")
					save_dict['status'] = "success"
					save_dict['found'] = found
					save_dict['target_grasp_pose'] = grasp_pose[0].cpu().numpy()
					save_dict['graspability'] = graspability.item()
					# save
					with open(file_name, 'wb') as f:
						pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
					break
			else:
				if self.realworld:
					pass
				elif self.recognition:
					pass
				else:
					Ts = Ts[:-1]
					parameters = parameters[:-1]

			# check status
			if not torch.is_tensor(graspability):
				graspability = torch.tensor(graspability)
			#print(f'found: {found}, graspable: {graspability.item()}')
			save_dict['found'] = found
			save_dict['graspability'] = graspability.item(),

			if iteration == self.max_iter:
				# save
				#print("arrived at max iter")
				save_dict['status'] = "failed by reaching max iteration"
				with open(file_name, 'wb') as f:
					pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
				break
   
			# check if objects are in workspace
			if not self.realworld:
				if self.recognition:
					is_in_workspace = self.check_objects_are_in_workspace(self.workspace_bounds.to(self.device), Ts_gt.to(self.device))
				else:
					is_in_workspace = self.check_objects_are_in_workspace(self.workspace_bounds.to(self.device), Ts.to(self.device))
				if not is_in_workspace:
					#print("objects out of workspace")
					save_dict['status'] = "failed by objects out of workspace"
					with open(file_name, 'wb') as f:
						pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
					break
   
			# #####################
			# target_pose = self.env.groundtruth_recognition(output_dtype='torch')[0][-1].to(self.device)
			# pc_of_target_object = get_pc_of_objects(target_pose.unsqueeze(0), self.target_sq.unsqueeze(0)).squeeze().to(self.device)
			# graspability, grasp_pose = check_graspability(
			# target_pose,
			# self.target_sq,
			# Ts[[]],
			# parameters[[]],
			# pc_of_target_object,
			# self.env.sim._get_shelf_info(),
			# self.gripper_open_pc,
			# visualize=True
			# )
			# #####################
   
   			# find best action
			actions = self.search_policy.find_best_action(
				Ts, parameters
			)
			
			# if policy fail
			if actions[0]['action_type'] == 'fail':
				#print(best_action['description'])
				save_dict['status'] = actions[0]['description']
				with open(file_name, 'wb') as f:
					pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
				if self.realworld:
					data_to_send['action_type'] = 'fail'
					server.send_grasp(data_to_send)
					server.close_connection()
				break
			
			# control

			control_success = False
			for action in actions:
				if self.realworld:
					control_success, list_qd, list_vel, list_acc = self.env.implement_action_realworld(action)
				else:
					control_success = self.env.implement_action(action)
				if not control_success:
					self.env.sim.reset_robot()
					self.env.sim.robot_go_home()
					if not self.realworld:
						self.env._reset_objects(Ts_gt[:,0:3,3].cpu().numpy(), matrices_to_quats_torch(Ts_gt[:,0:3,0:3]).cpu().numpy())
				else:
					if self.realworld:
						break
					else:
						Ts_gt_temp, _ = self.env.groundtruth_recognition(output_dtype='torch')
						is_in_workspace_temp = self.check_objects_are_in_workspace(self.workspace_bounds.to(self.device), Ts_gt_temp.to(self.device))
						if not is_in_workspace_temp:
							self.env._reset_objects(Ts_gt[:,0:3,3].cpu().numpy(), matrices_to_quats_torch(Ts_gt[:,0:3,0:3]).cpu().numpy())
						else:
							break

			#print(f"control success : {control_success}")
			if not control_success:
				save_dict['status'] = "failed to control"
				with open(file_name, 'wb') as f:
					pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
				if self.realworld:
					data_to_send['action_type'] = action['action_type']
					data_to_send['status'] = 'fail'
					server.send_grasp(data_to_send)
					server.close_connection()
				break
			elif control_success and self.realworld:
				data_to_send['action_type'] = action['action_type']
				data_to_send['status'] = 'success' 			
				data_to_send['list_qd'] = list_qd
				data_to_send['list_vel'] = list_vel
				data_to_send['list_acc'] = list_acc
				server.send_grasp(data_to_send)
				server.close_connection()
	
			# save action
			action_numpy = deepcopy(action)
			for key in action_numpy:
				if torch.is_tensor(action_numpy[key]):
					action_numpy[key] = action_numpy[key].cpu().numpy()
			save_dict['implemented_action'] = action_numpy
			# save data
   
			save_dict['status'] = "normal operation"
			with open(file_name, 'wb') as f:
				pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)

			# iteration number
			iteration += 1
		print(save_dict['status'])
		return True

	#############################################################
	##################### RECOGNITION CODES #####################
	#############################################################

	def simulation_recognition(self, pc, label):
		
		# initialize
		data = {}
		data['pc'] = pc
		data['seg_pred'] = label
		data['mean_xyz'] = torch.zeros(3,1).unsqueeze(0)
		data['diagonal_len'] = torch.tensor(1).unsqueeze(0)

		# forward
		data = self.sq_recognition(data)
		data = self.unnormalize(data)
		
		# processing
		Ts = data['Ts_pred'].squeeze(0)
		parameters = data['parameters'].squeeze(0)

		return Ts, parameters

	def realworld_recognition(self, pc, labels=None):
			
		# initialize
		data = {}
		data['pc'] = pc

		# segmentation
		if labels is not None:
			data['seg_pred'] = labels
			data['mean_xyz'] = torch.zeros(3,1).unsqueeze(0)
			data['diagonal_len'] = torch.tensor(1).unsqueeze(0)
		else:
			pc_input, mean_xyz_global, diagonal_len_global = normalize_pointcloud_seg(pc.squeeze(0).cpu().numpy())
			data['pc'] = torch.Tensor(pc_input).unsqueeze(0)
			data['mean_xyz'] = torch.Tensor(mean_xyz_global).unsqueeze(0)
			data['diagonal_len'] = torch.tensor(diagonal_len_global).unsqueeze(0)
			data = self.segmentation(data)
			data = self.calibration(data)

		# forward
		data = self.sq_recognition(data)
		data = self.unnormalize(data)
		
		# processing
		Ts = data['Ts_pred'].squeeze(0)
		parameters = data['parameters'].squeeze(0)

		return Ts, parameters

	def segmentation(self, data):
		# test
		pc = data['pc'].to(self.device)

		seg_pred = self.segmentation_model(pc)
		seg_pred = seg_pred.detach()

		data['pc'] = pc
		data['seg_pred'] = seg_pred

		return data

	def calibration(self, data):
		pc_batch = data['pc']
		seg_pred_batch = data['seg_pred']

		num_classes = seg_pred_batch.shape[2]

		for batch_idx, (pc, seg_pred) in enumerate(zip(pc_batch, seg_pred_batch)):
			pc = pc.cpu().numpy().T
			seg_pred = seg_pred.argmax(axis=1)

			pcd = o3d.geometry.PointCloud()
			pcd.points = o3d.utility.Vector3dVector(pc)
			partial_pcd_tree = o3d.geometry.KDTreeFlann(pcd)

			seg_pred_raw = torch.clone(seg_pred)

			for i in range(len(seg_pred_raw)):
				[_, idxs, _] = partial_pcd_tree.search_knn_vector_3d(pcd.points[i], self.calibration_k)
				seg_pred_near = seg_pred_raw[idxs]
				seg_pred_near = seg_pred_near[seg_pred_near != 0]
				seg_pred[i] = torch.bincount(seg_pred_near).argmax() if len(seg_pred_near) else seg_pred[i]

			temp_torch_eye = torch.eye(num_classes).to(self.device) 
			seg_pred_1hot = temp_torch_eye[seg_pred]

			seg_pred_batch[batch_idx] = seg_pred_1hot

		data['seg_pred'] = seg_pred_batch.argmax(dim=2)

		return data

	def sq_recognition(self, data):
		pc_batch = data['pc']
		seg_pred_batch = data['seg_pred']

		batch_size = len(pc_batch)
		for seg_pred in seg_pred_batch:
			seg_ids, seg_counts = torch.unique(seg_pred, return_counts=True)
			seg_valid_ids = torch.where(seg_counts > self.num_pts_recog)[0].cpu().tolist()
			max_num_primitives = len(seg_valid_ids)

		# max_num_primitives = max([len(torch.unique(seg_pred)) for seg_pred in seg_pred_batch])
		recog_preds_batch = torch.zeros(batch_size, max_num_primitives, self.recognition_model.output_dim_total).to(self.device)
		mean_xyz_pris_batch = torch.zeros(batch_size, max_num_primitives, 3, 1)
		diagonal_len_pris_batch = torch.zeros(batch_size, max_num_primitives)

		# seg_ids_batch = torch.zeros(batch_size, self.seg_module.num_classes)
		num_primitives_batch = torch.zeros(batch_size, dtype=int)
		
		for batch_idx in range(len(pc_batch)):
			pc_scene = pc_batch[batch_idx]
			seg_pred = seg_pred_batch[batch_idx]

			seg_ids, seg_counts = torch.unique(seg_pred, return_counts=True)
			seg_valid_ids = seg_ids[torch.where(seg_counts > self.num_pts_recog)[0]].cpu().tolist()
			# seg_ids_batch[batch_idx, :len(seg_ids)] = seg_ids
			num_primitives_batch[batch_idx] = len(seg_valid_ids)

			pc = []

			for primitive_idx, seg_id in enumerate(seg_valid_ids):
				# get primitive-wise point cloud
				pc_pri = pc_scene[:, seg_pred == seg_id]
				pc_pri = torch.cat([pc_pri, torch.ones(1, pc_pri.shape[1]).to(self.device)])

				pc_surround = pc_scene[:, seg_pred != seg_id]
				pc_surround = torch.cat([pc_surround, torch.zeros(1, pc_surround.shape[1]).to(self.device)])

				pc_overall = torch.cat([pc_pri, pc_surround], dim=1)

				# normalize point cloud
				pc_overall, mean_xyz_pri, diagonal_len_pri = normalize_pointcloud_recog(pc_overall.cpu().numpy(), pc_pri.cpu().numpy())

				pc += [torch.Tensor(pc_overall)]

				mean_xyz_pris_batch[batch_idx, primitive_idx] = torch.Tensor(mean_xyz_pri)
				diagonal_len_pris_batch[batch_idx, primitive_idx] = diagonal_len_pri.item()

			pc = torch.stack(pc, dim=0).to(self.device)

			# test
			recog_preds = self.recognition_model(pc)
			recog_preds = recog_preds.detach()

			recog_preds_batch[batch_idx] = recog_preds

		data['recog_preds'] = recog_preds_batch
		# data['seg_ids_batch'] = seg_ids_batch
		data['num_primitives_batch'] = num_primitives_batch
		data['mean_xyz_pris'] = mean_xyz_pris_batch
		data['diagonal_len_pris'] = diagonal_len_pris_batch

		return data

	def unnormalize(self, data):
		pc = data['pc'].permute([0, 2, 1])
		recog_preds = data['recog_preds']
		mean_xyz_objects = data['mean_xyz_pris'].to(self.device).squeeze(-1)
		diagonal_len_objects = data['diagonal_len_pris'].to(self.device).unsqueeze(-1)
		mean_xyz_global = data['mean_xyz'].to(self.device).permute([0, 2, 1])
		diagonal_len_global = data['diagonal_len'].to(self.device).unsqueeze(-1).unsqueeze(-1)

		# decompose output
		positions = deepcopy(recog_preds[:, :, :3])
		orientations = deepcopy(recog_preds[:, :, 3:7])
		parameters = deepcopy(recog_preds[:, :, 7:])

		# revise position and parameters primitive-wisely
		positions = positions * diagonal_len_objects + mean_xyz_objects
		parameters[:, :, :3] *= diagonal_len_objects

		# revise position and parameters globaly
		pc = pc * diagonal_len_global + mean_xyz_global
		positions = positions * diagonal_len_global + mean_xyz_global
		parameters[:, :, :3] *= diagonal_len_global

		# get before-action-SE3 predictions
		Rs = quats_to_matrices_torch(orientations.reshape(-1, orientations.shape[2]))
		Ts = get_SE3s_torch(Rs, positions.reshape(-1, positions.shape[2]))
		Ts = Ts.reshape(recog_preds.shape[0], recog_preds.shape[1], Ts.shape[1], Ts.shape[2])

		data['pc'] = pc.permute([0, 2, 1])
		data['Ts_pred'] = Ts
		data['parameters'] = parameters

		return data

	def load_segmentation_network(self):
		model_root = 'pretrained/segmentation_config'
		model_identifier = 'pretrained'
		model_config_file = 'segmentation_config.yml'
		model_ckpt_file = 'model_best.pkl'
		model, _ = load_pretrained(
			identifier=model_identifier,
			config_file=model_config_file,
			ckpt_file=model_ckpt_file,
			root=model_root
		)
		return model.to(self.device)

	def load_recognition_network(self):
		model_root = 'pretrained/recognition_config'
		model_identifier = 'pretrained'
		model_config_file = 'recognition_config.yml'
		model_ckpt_file = 'model_best.pkl'
		model, _ = load_pretrained(
			identifier=model_identifier,
			config_file=model_config_file,
			ckpt_file=model_ckpt_file,
			root=model_root
		)
		return model.to(self.device)

	#############################################################
	####################### SPAWN OBJECTS #######################
	#############################################################

	def spawn_target_object(self):
		Ts, parameters = self.env.groundtruth_recognition(
			output_dtype='torch'
		)
		fully_occluded = get_pose_map(
			self.pose_map_grid,
			self.target_sq,
			Ts.to(self.device),
			parameters.to(self.device),
			self.depth_renderer
		)

		# # rendering pose map and target object
		# render_map(
		# 	Ts.to(self.device), 
		# 	parameters.to(self.device),
		# 	self.pose_map_grid,
		# 	fully_occluded_positions, 
		# 	self.shelf_info
		# 	)

		# spawn target object
		target_pose = get_poses_from_grid(self.pose_map_grid[fully_occluded], self.target_sq[2])
		spawn_success = False
		for i in torch.randperm(len(target_pose)):
			pc_of_target_object = get_pc_of_objects(target_pose[i].unsqueeze(0), self.target_sq.unsqueeze(0)).squeeze().to(self.device)
			#print(target_pose[i])
			graspable, _ = check_graspability(
						target_pose[i].to(self.device),
						self.target_sq.to(self.device),
						Ts[[]].to(self.device),
						parameters[[]].to(self.device),
						pc_of_target_object,
						self.shelf_info,
						self.gripper_open_pc.to(self.device),
						visualize=False
						)

			if graspable:
				#print("spawn target")
				if self.target_object == 'cylinder':
					self.env._create_cylinder(
						self.target_sq[0].cpu().numpy(),
						self.target_sq[2].cpu().numpy() * 2.0,
						target_pose[i, 0:3, 3].cpu().numpy(),
						[0, 0, 0],
						scale_cylinder=1,
						object_color = (153/255, 0.0, 17/255, 1.0)
					)
				elif self.target_object == 'box':
					#print(self.pose_map_grid[fully_occluded][i, 3].cpu().numpy())
					self.env._create_box(
						self.target_sq[0].cpu().numpy() * 2.0,
						self.target_sq[1].cpu().numpy() * 2.0,
						self.target_sq[2].cpu().numpy() * 2.0,
						target_pose[i, 0:3, 3].cpu().numpy(),
						[0, 0, self.pose_map_grid[fully_occluded][i, 3].cpu().numpy()],
						scale_box=1,
						object_color = (153/255, 0.0, 17/255, 1.0)
					)
				_, _, _, _, mask = self.env.observe() 
				found_init = (mask == self.num_objects + 4).sum() > 100
				if found_init:
					#print('target object is seen at initial position')
					temp_object_id = self.num_objects + 4
					p.removeBody(temp_object_id)
					self.env.object_ids = self.env.object_ids[:-1]
					self.env.object_infos = self.env.object_infos[:-1]
					continue

				spawn_success = True
				break

		return spawn_success

	#############################################################
	############################ etc ############################
	#############################################################	
 
	def check_objects_are_in_workspace(self, workspace_bounds, Ts):
		workspace_bounds = workspace_bounds.to(self.device)
		positions = Ts[:, 0:3, 3]
		bools = (workspace_bounds[:,0] - 0.1 < positions) & (positions < workspace_bounds[:,1] + 0.1)
		return bools.all()


def get_poses_from_grid(grid, height_offset=0.):
    if grid.shape[-1] == 3:
        SE3s = torch.eye(4).repeat(len(grid), 1, 1).to(grid.device)
        SE3s[:, 0:3, 3] = grid # add positions in grid
        SE3s[:, 2, 3] += height_offset # lift up target object by its height
    elif grid.shape[-1] == 4:
        SE3s = torch.eye(4).repeat(len(grid), 1, 1).to(grid.device)
        SE3s[:, 0:3, 3] = grid[:,0:3] # add positions in grid
        w = torch.zeros_like(grid[:,0:3])
        w[:,2] = grid[:,3]
        SE3s[:, 0:3, 0:3] = exp_so3(w)
        SE3s[:, 2, 3] += height_offset # lift up target object by its height

    return SE3s