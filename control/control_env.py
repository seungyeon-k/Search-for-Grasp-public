import pybullet as p
import os
import open3d as o3d
import numpy as np
import torch
import time
import random
import torch
from data_generation.sim_with_shelf import PybulletShelfSim
from data_generation.utils import project_pts_to_3d, euler2rotm, project_pts_to_2d
import pickle
from matplotlib import colormaps
from copy import deepcopy

from functions.superquadrics import get_SQ_parameters
from functions.lie import quats_to_matrices, get_SE3s, matrices_to_quats

class ControlSimulationEnv:
	def __init__(self, enable_gui=True, blender_recorder=False):
			
		# pybullet settings
		self.enable_gui = enable_gui
		self.sim = PybulletShelfSim(
			enable_gui=enable_gui,
			blender_recorder=blender_recorder
			)
		self.blender_recorder = blender_recorder

		# environment settings
		self.workspace_bounds = self.sim.workspace_bounds
		self.num_pts_down_wo_plane = 2048
		self.spawn_bounds = self.sim.spawn_bounds
		self.voxel_size = 0.004
		self.box_size_random_min = np.array([2, 4, 2]) * 5 
		self.box_size_random_max = np.array([6, 10, 15]) * 5 
		self.cylinder_size_random_min = np.array([1.0, 4.5]) * 5 
		self.cylinder_size_random_max = np.array([3.0, 15]) * 5 
		self.box_size_list_dict = {}
		self.cylinder_size_list_dict = {}
		self.box_size_list_dict['known'] = np.array([[5.15, 2.9, 3.85], # SPAM
													 [4.25, 1.75, 3.5], # Jell-O
													 [3.05, 1.25, 6], # yellow box
													 [5.35, 2.5, 5.4], # Oat cookie
													 [4.75, 2.75, 8], # Raspberry cookie
						   							 [7.75, 2.2, 9.65], # Cheeze-it
			  										]) * 5 * 2
		self.cylinder_size_list_dict['known'] = np.array([[3.4, 21.4], # pringles
														  [3.3, 13.4], # cantata
														  [3.3, 12.2], # yamyam
														  [2.5, 13.4], # Hotsix
														  [2.8, 6], # Wood
														  ]) * 5
		self.spawn_range = np.array([0.3, 0.7])
		self.object_ids = []
		self.object_infos = []

	#############################################################
	################## ACTION IMPLEMENTATION ####################
	#############################################################

	def implement_action(self, best_action):
		
		if best_action['action_type'] == 'grasping':
			initial_grasp_pose = best_action['initial_grasp_pose']
			final_grasp_pose = best_action['final_grasp_pose']
			
			return self.sim.pick_and_place(
						initial_grasp_pose.cpu().numpy(),
						final_grasp_pose.cpu().numpy()
					)
		elif best_action['action_type'] == 'pushing':
			initial_push_pose = best_action['initial_push_pose']
			final_push_pose = best_action['final_push_pose']

			return self.sim.pushing(
						initial_push_pose.cpu().numpy(),
						final_push_pose.cpu().numpy()
					)
		elif best_action['action_type'] == 'target_retrieval':
			grasp_pose = best_action['grasp_pose']
			return self.sim.target_retrieval(
						grasp_pose.cpu().numpy(),
					)

	def implement_action_realworld(self, best_action):
			
		if best_action['action_type'] == 'grasping':
			initial_grasp_pose = best_action['initial_grasp_pose']
			final_grasp_pose = best_action['final_grasp_pose']
			
			return self.sim.pick_and_place_realworld(
						initial_grasp_pose.cpu().numpy(),
						final_grasp_pose.cpu().numpy(),
					)
		elif best_action['action_type'] == 'pushing':
			initial_push_pose = best_action['initial_push_pose']
			final_push_pose = best_action['final_push_pose']

			return self.sim.pushing_realworld(
						initial_push_pose.cpu().numpy(),
						final_push_pose.cpu().numpy()
					)
		elif best_action['action_type'] == 'target_retrieval':
			grasp_pose = best_action['grasp_pose']
			return self.sim.target_retrieval_realworld(
						grasp_pose.cpu().numpy(),
					)


	#############################################################
	################# ENVIRONMENT INITIALIZE ####################
	#############################################################

	def start_simulation(self):
		self.sim.start_simulation_thread()

	def reset(self, cfg_objects):

		while True:

			# remove objects
			for obj_id in self.object_ids:
				p.removeBody(obj_id)

			# initialize
			self.object_ids = []
			self.object_infos = []

			# load objects
			if cfg_objects.spawn_mode == 'random':
				self._random_drop(num_objects=cfg_objects.num_objects)
				time.sleep(1)
			elif cfg_objects.spawn_mode == 'load':
				self.load_objects(cfg_objects.obj_file_dir)
			else:
				raise NotImplementedError
			
			# wait until objects stop moving
			flag = False
			old_po = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
			for _ in range(10):
				time.sleep(1)
				new_ps = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				if np.sum((new_ps - old_po) ** 2) < 1e-6:
					flag = True
					break
				old_po = new_ps
			if not flag:
				continue

			# update object info
			for i, obj_id in enumerate(self.object_ids):
				self.object_infos[i]['position'] = p.getBasePositionAndOrientation(obj_id)[0]
				self.object_infos[i]['orientation'] = p.getBasePositionAndOrientation(obj_id)[1]

			return

	def _random_drop(
			self, 
			object_types=['cylinder', 'box'],
			knowledge='known',
			num_objects=8,
			enable_stacking=False,
   			enable_trip=False,
	  		save=False,
			save_dir=None
		):
		# object_types=['box', 'cylinder'], 
		# num_objects=8, 

		# enable stacking
		self.enable_stacking = enable_stacking
  
		# enable tripping
		self.enable_trip = enable_trip
		
		# set distance threshold
		distance_threshold = 0.02 if not enable_stacking else 0.05

		# sample object positions
		while True:
			xy_pos = np.random.rand(num_objects, 2)
			xy_pos = (
				np.expand_dims(self.spawn_bounds[:2, 0], 0)
				+ np.expand_dims(
					self.spawn_bounds[:2, 1] - self.spawn_bounds[:2, 0], 0
				) * xy_pos
			)
   
			if num_objects == 1:
				break

			distance_list = []
			for i in range(num_objects - 1):
				for j in range(i + 1, num_objects):
					distance = np.sqrt(np.sum((xy_pos[i] - xy_pos[j])**2))
					distance_list += [distance]

			# if not enable stacking, make objects far away
			if not enable_stacking and min(distance_list) > distance_threshold:
				break

			# if enable stacking, make objects close each others
			if enable_stacking and max(distance_list) < distance_threshold:
				break

		# make all objects locate around the center of workspce

		for i in range(num_objects):
			
			# set object type
			if enable_stacking:
				object_type = 'box'
			else:
				object_type = random.choice(object_types)

			# put object in simulator
			if object_type == 'box':
					
				# scale
				scale_box = 500
				
				# choose box in discrete set / continuous set
				if knowledge in ['known', 'unknown']:
					box_index = np.random.randint(
						len(self.box_size_list_dict[knowledge])
					)
					size_box = self.box_size_list_dict[knowledge][box_index]
				elif knowledge == 'random':
					size_box = (self.box_size_random_max - self.box_size_random_min) * np.random.rand(len(self.box_size_random_min)) + self.box_size_random_min 
				else:
					raise ValueError('check knowledge variable')
				# rotating box
				exchange_dims = random.choice(
					#[[0, 1, 2], [1, 2, 0], [2, 0, 1]]
					[[0, 1, 2], [1, 0, 2]]
				)
				size_box = size_box[exchange_dims]
				size_x = (np.round(size_box[0])).astype(int)
				size_y = (np.round(size_box[1])).astype(int)
				size_z = (np.round(size_box[2])).astype(int)

				# object pose
				if not enable_stacking or not enable_trip:
					orientation = [0, 0, np.random.rand()*2*np.pi]
				else:
					orientation = [
						np.random.rand()*2*np.pi, 
						np.random.rand()*2*np.pi, 
						np.random.rand()*2*np.pi
					]
				position = np.append(
					xy_pos[i], 
					size_z / scale_box / 2 + self.workspace_bounds[2, 0] + 0.01
				)
				if enable_stacking:
					position[2] += 0.1
				
				# create box
				self._create_box(
					size_x, 
					size_y, 
					size_z, 
					position, 
					orientation, 
					scale_box=scale_box,
					enable_stacking=enable_stacking
				)

				# time sleep
				time.sleep(0.2 if not enable_stacking else 0.5)

			elif object_type == 'cylinder':

				# scale
				scale_cylinder = 500

				# choose cylinder in discrete set / continuous set
				if knowledge in ['known', 'unknown']:
					cylinder_index = np.random.randint(
						len(self.cylinder_size_list_dict[knowledge])
					)
					size_cylinder = self.cylinder_size_list_dict[knowledge][cylinder_index]
				elif knowledge == 'random':
					size_cylinder = (self.cylinder_size_random_max - self.cylinder_size_random_min) * np.random.rand(len(self.cylinder_size_random_min)) + self.cylinder_size_random_min 
				else:
					raise ValueError('check knowledge variable')

				# declare cylinder
				size_r = (np.round(size_cylinder[0])).astype(int)
				size_h = (np.round(size_cylinder[1])).astype(int)

				# object pose
				position = np.append(
					xy_pos[i], 
					self.workspace_bounds[2, 0] + size_h / scale_cylinder / 2 + 0.01
				)
				orientation = [0, 0, np.random.rand()*2*np.pi]
				
				# create object
				self._create_cylinder(
					size_r, 
					size_h, 
					position, 
					orientation,
					scale_cylinder=scale_cylinder
				)
				
				# time sleep
				time.sleep(0.2)
	
			if save:
				with open(save_dir,"wb") as f:
					pickle.dump(self.object_infos, f)

	def load_objects(self, file_dir):
		with open(file_dir,"rb") as f:
			object_infos = pickle.load(f)
		self.object_infos = []
		for self.loaded_object_idx, object_info in enumerate(object_infos):
			type = object_info["type"]
			size = object_info['size']
			position = object_info['position']
			orientation = object_info['orientation']
			object_color = object_info['object_color']

			# create object
			if type == 'box':
				self._create_box(
					size[0], 
					size[1], 
					size[2], 
					position, 
					orientation, 
					scale_box=1,
					object_color=object_color
				)
			elif type == 'cylinder':
				self._create_cylinder(
					size[0], 
					size[2], 
					position, 
					orientation,
					scale_cylinder=1,
					object_color=object_color
	 			)
    
	def _reset_objects(self, old_pos, old_ors):
		for idx, object_id in enumerate(self.object_ids):
			p.resetBasePositionAndOrientation(object_id, old_pos[idx], old_ors[idx])	 
   
	#############################################################
	################ GROUND-TRUTH RECOGNITION ###################
	#############################################################

	def groundtruth_recognition(self, output_dtype='numpy'):

		# positions and orientations
		po_ors = [p.getBasePositionAndOrientation(object_id) for object_id in self.object_ids]
		positions = np.array([po_or[0] for po_or in po_ors])
		orientations = np.array([po_or[1] for po_or in po_ors])
		rotations = quats_to_matrices(orientations)
		Ts = get_SE3s(rotations, positions)
		if output_dtype == 'numpy':
			pass
		elif output_dtype == 'torch':
			Ts = torch.tensor(Ts).float()
		
		# shape paremters
		shapes, sizes = get_SQ_parameters(self.object_infos)
		parameters = np.concatenate(
			[sizes, shapes], 
			axis=1
		)
		if output_dtype == 'numpy':
			pass
		elif output_dtype == 'torch':
			parameters = torch.tensor(parameters).float()

		return Ts, parameters

	#############################################################
	######################### COMMON ############################
	#############################################################

	def create_object_form_object_info(self, object_info, goal_pose):
			
		# object information
		object_type = object_info['type']
		object_size = object_info['size']

		# object pose
		goal_pose = goal_pose.cpu().numpy()
		position = goal_pose[:3, 3]
		orientation = matrices_to_quats(goal_pose[:3, :3])

		# create object
		if object_type == 'box':
			self._create_box(
				object_size[0], 
				object_size[1], 
				object_size[2], 
				position, 
				orientation, 
				scale_box=1,
			)
		elif object_type == 'cylinder':
			self._create_cylinder(
				object_size[0], 
				object_size[2], 
				position, 
				orientation,
				scale_cylinder=1
			)

	def _create_box(
			self, 
			size_x, 
			size_y, 
			size_z, 
			position, 
			orientation, 
			scale_box=500, 
			enable_stacking=False,
			object_color=None
		):

		# orientation
		orientation = p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation

		# color info
		if object_color is None:
			object_color = colormaps['Blues'](np.random.rand() * 0.7 + 0.3)

		# declare objects
		if self.blender_recorder:
				
			# save directory
			urdf_folder = 'assets/shelf_objects'
			mesh_folder = os.path.join(urdf_folder, 'meshes')
			if not os.path.exists(mesh_folder):
				os.makedirs(mesh_folder)	
					
			# paths
			urdf_path = os.path.join(
				urdf_folder, 
				f'object_{self.loaded_object_idx}.urdf'
			)
			mesh_path = os.path.join(
				mesh_folder, 
				f'object_{self.loaded_object_idx}.obj'
			)
			filename = os.path.join(
				'meshes',
				f'object_{self.loaded_object_idx}.obj'
			)

			# save mesh
			mesh_box = o3d.geometry.TriangleMesh.create_box(
				width = size_x, 
				height = size_y, 
				depth = size_z
			)
			mesh_box.translate([-size_x/2, -size_y/2, -size_z/2])
			mesh_box.paint_uniform_color([object_color[0], object_color[1], object_color[2]])
			o3d.io.write_triangle_mesh(
				mesh_path, 
				mesh_box,
				# write_vertex_colors=True
			)

			# urdf contents
			urdf_contents = f"""
			<robot name="object_{self.loaded_object_idx}">
				<link name="link_object_{self.loaded_object_idx}">
					<visual>
						<geometry>
							<mesh filename="{filename}" />
						</geometry>
					</visual>

					<collision>
						<geometry>
							<mesh filename="{filename}" />
						</geometry>
					</collision>
					
					<inertial>
						<origin rpy="0 0 0" xyz="0 0 0"/>
						<mass value="0.3" />
						<inertia ixx="0.3" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.3" />
					</inertial>
				</link>
			</robot>
			"""

			# save
			file = open(urdf_path, "w")
			file.write(urdf_contents)
			file.close()
			time.sleep(0.2)

			# load object
			body_id = p.loadURDF(
				urdf_path, 
				position, 
				orientation
			)

		else:

			# declare collision
			collision_id = p.createCollisionShape(
				p.GEOM_BOX, 
				halfExtents=np.array(
					[size_x/2, size_y/2, size_z/2]
				) / scale_box
			)
			
			# create object
			body_id = p.createMultiBody(
				0.05, 
				collision_id, 
				-1, 
				position, 
				orientation
			)

		p.changeDynamics(
			body_id, 
			-1, 
			spinningFriction=0.002 if not enable_stacking else 0.2, 
			lateralFriction=0.4 if not enable_stacking else 0.6, 
			mass=0.737*size_x*size_y*size_z/(scale_box**3/1000)
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=object_color
		)
		self.object_ids.append(body_id)

		# keep object information
		object_info = {
			'type': 'box', 
			'size': [size_x/scale_box, size_y/scale_box, size_z/scale_box],
			'position': position, 
			'orientation': orientation,
			'object_color': object_color
		}
		self.object_infos.append(object_info)

		# blender 
		if self.blender_recorder:
			self.sim.sim_recorder_register_object(
				body_id, 
				urdf_path, 
				global_color=object_color
			)

		return body_id

	def _create_cylinder(
			self, 
			size_r, 
			size_h, 
			position, 
			orientation, 
			scale_cylinder=500,
			object_color=None
		):

		# orientation
		orientation = p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation

		# color info
		if object_color is None:
			object_color = colormaps['Blues'](np.random.rand() * 0.7 + 0.3)

		# declare objects
		if self.blender_recorder:
				
			# save directory
			urdf_folder = 'assets/shelf_objects'
			mesh_folder = os.path.join(urdf_folder, 'meshes')
			if not os.path.exists(mesh_folder):
				os.makedirs(mesh_folder)	
					
			# paths
			urdf_path = os.path.join(
				urdf_folder, 
				f'object_{self.loaded_object_idx}.urdf'
			)
			mesh_path = os.path.join(
				mesh_folder, 
				f'object_{self.loaded_object_idx}.obj'
			)
			filename = os.path.join(
				'meshes',
				f'object_{self.loaded_object_idx}.obj'
			)

			# save mesh
			mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
				radius=size_r, 
				height=size_h,
				resolution=50,
				split=4
			)
			
			mesh_cylinder.paint_uniform_color([object_color[0], object_color[1], object_color[2]])
			o3d.io.write_triangle_mesh(
				mesh_path, 
				mesh_cylinder,
				# write_vertex_colors=True
			)
			
			# urdf contents
			urdf_contents = f"""
			<robot name="object_{self.loaded_object_idx}">
				<link name="link_object_{self.loaded_object_idx}">
					<visual>
						<geometry>
							<mesh filename="{filename}" />
						</geometry>
					</visual>

					<collision>
						<geometry>
							<mesh filename="{filename}" />
						</geometry>
					</collision>
					
					<inertial>
						<origin rpy="0 0 0" xyz="0 0 0"/>
						<mass value="0.3" />
						<inertia ixx="0.3" ixy="0" ixz="0" iyy="0.3" iyz="0" izz="0.3" />
					</inertial>
				</link>
			</robot>
			"""

			# save
			file = open(urdf_path, "w")
			file.write(urdf_contents)
			file.close()
			time.sleep(0.2)

			# load object
			body_id = p.loadURDF(
				urdf_path, 
				position, 
				orientation
			)

		else:

			# declare collision
			collision_id = p.createCollisionShape(
				p.GEOM_CYLINDER, 
				radius=size_r/scale_cylinder, 
				height=size_h/scale_cylinder
			)
			
			# create object
			body_id = p.createMultiBody(
				0.05, 
				collision_id, 
				-1, 
				position, 
				orientation
			)

		p.changeDynamics(
			body_id, 
			-1, 
			spinningFriction=0.2, 
			lateralFriction=0.6, 
			mass=0.585*np.pi*size_r*size_r*size_h/(scale_cylinder**3/1000)
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=object_color
		)
		self.object_ids.append(body_id)

		# keep object information
		object_info = {'type': 'cylinder',
				 	   'size': [size_r/scale_cylinder, size_r/scale_cylinder, size_h/scale_cylinder],
					   'position': position, 
					   'orientation': orientation,
					   'object_color' : object_color,
					  }
		self.object_infos.append(object_info)

		# blender 
		if self.blender_recorder:
			self.sim.sim_recorder_register_object(
				body_id, 
				urdf_path, 
				global_color=object_color
			)

		return body_id

	#############################################################
	#################### VISION SENSOR ##########################
	#############################################################

	def observe(self):
		color_image, depth_image, mask_image = self.sim.get_camera_data(self.sim.camera_params[0])

		camera_pose = self.sim.camera_params[0]['camera_pose']
		camera_intr = self.sim.camera_params[0]['camera_intr']

		organized_pc, _ = project_pts_to_3d(color_image, depth_image, camera_intr, camera_pose)

		# pc = self._get_workspace_pc(organized_pc)
		pc = organized_pc.reshape(-1, organized_pc.shape[2])
		labels = mask_image.reshape(-1)

		wo_plane_idxs = (
			(pc[:, 0] >= self.workspace_bounds[0, 0])
			* (pc[:, 0]<=self.workspace_bounds[0, 1])
			* (pc[:, 1]>=self.workspace_bounds[1, 0])
			* (pc[:, 1]<=self.workspace_bounds[1, 1])
			* (pc[:, 2]>=self.workspace_bounds[2, 0] + 0.002)
			* (pc[:, 2]<=self.workspace_bounds[2, 1])
		)
		wo_plane_idxs = np.where(wo_plane_idxs==1)[0].tolist()

		down_wo_plane_idxs = random.sample(wo_plane_idxs, self.num_pts_down_wo_plane) if len(wo_plane_idxs) > self.num_pts_down_wo_plane else wo_plane_idxs
		pc = pc[down_wo_plane_idxs]
		labels = labels[down_wo_plane_idxs]

		return pc, labels, color_image, depth_image, mask_image

	def _get_workspace_pc(self, organized_pc):
		pc = organized_pc.reshape(-1, organized_pc.shape[2])
				
		valid_idxs = np.logical_and(
			np.logical_and(
				np.logical_and(pc[:, 0]>=self.workspace_bounds[0, 0], pc[:, 0]<=self.workspace_bounds[0, 1]),
				np.logical_and(pc[:, 1]>=self.workspace_bounds[1, 0], pc[:, 1]<=self.workspace_bounds[1, 1])
			),
			np.logical_and(pc[:, 2]>=self.workspace_bounds[2, 0]-0.001, pc[:, 2]<=self.workspace_bounds[2, 1])
		)

		pc = pc[valid_idxs]

		return pc
