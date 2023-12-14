import time
import numpy as np
import pybullet as p
import random
import open3d as o3d
import math
from copy import deepcopy

from .sim_with_shelf import PybulletShelfSim
from functions.point_clouds import upsample_pointcloud
from .utils import euler2rotm, project_pts_to_2d, project_pts_to_3d, get_volume
from functions.utils import get_SE3s, quats_to_matrices
import matplotlib.pyplot as plt

from policy.push_planner import push_planner

class SimulationShelfEnv():
	def __init__(self, enable_gui):
		self.enable_gui = enable_gui
		self.sim = PybulletShelfSim(enable_gui=enable_gui)
		self.workspace_bounds = self.sim.workspace_bounds
		self.spawn_bounds = self.sim.spawn_bounds
		self.num_pts_down = 4096
		self.num_pts_down_wo_plane = 2048
		self.num_pts_gt = 512
		self.num_directions = 8
		self.num_z = 5
		self.voxel_size = 0.004
		self.object_ids = []

		# box size list (cm)
		self.box_size_list_dict = {}
		self.box_size_list_dict['known'] = np.array([[3, 7.5, 8.5],
													 [5.5, 5.5, 5.5],
													 [10.5, 15, 5.5],
													 [9.5, 5.5, 16],
													 [2.5, 15, 19],
													 [12, 2.5, 6],
													 [5.8, 6.3, 6],
													 [6.4, 6, 2.8],
													 [6, 6.2, 5.8]]) * 5
		self.box_size_list_dict['unknown'] = np.array([[4, 10, 11.2],
													   [7, 7, 7],
													   [7.5, 11, 4],
													   [7, 4, 11.5],
													   [2, 10, 13],
													   [9.5, 2, 5],
													   [7.7, 8.4, 8],
													   [7.5, 7, 3.3],
													   [4, 4.1, 3.9]]) * 5
		self.box_size_random_min = np.array([1.25, 1.25, 1.25]) * 5 * 2
		self.box_size_random_max = np.array([9, 9, 9]) * 5 * 2

		# cylinder size list (cm)
		self.cylinder_size_list_dict = {}
		self.cylinder_size_list_dict['known'] = np.array([[3, 6],
														  [3, 10],
														  [2.5, 13],
														  [3.5, 20],
														  [4, 8],
														  [4, 4],
														  [4, 12],
														  [2, 3],
														  [4, 15]]) * 5
		self.cylinder_size_list_dict['unknown'] = np.array([[4.5, 9],
															[4, 13],
															[2, 10],
															[2.5, 14],
															[2, 4],
															[3, 3],
															[4.5, 13.5],
															[3, 4.5],
															[3, 12],]) * 5
		self.cylinder_size_random_min = np.array([2.5, 6]) * 5
		self.cylinder_size_random_max = np.array([4, 21]) * 5

	def reset(
			self, 
			object_types, 
			knowledge, 
			num_objects, 
			enable_stacking
		):

		# old position orientation for transform
		self.old_po_ors_for_transform = None

		while True:
			# remove objects
			for obj_id in self.object_ids:
				p.removeBody(obj_id)

			# object initialize
			self.object_ids = []
			self.voxel_coord = {}
			self.meshes = {}
			self.object_info = []
			self.vis_ratios = None

			# load objects
			self._random_drop(
				object_types, 
				knowledge, 
				num_objects, 
				enable_stacking
			)

			# wait until objets stop moving
			flag = False
			old_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
			for _ in range(50):
				time.sleep(0.1)
				new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				if np.sum((new_pos - old_pos) ** 2) < 1e-6:
					flag = True
					break
				old_pos = new_pos
			if not flag:
				continue

			# check if all objects are in workspace
			if not self._check_workspace():
				continue

			# check stacked if enable_stacking is False
			if not enable_stacking and self._check_stacked():
				continue

			# # check occlusion
			# if self._check_occlusion():
			# 	continue

			return

	def poke(self, only_recognition=False):
		output = {}

		# object information
		output['object_info'] = str(self.object_info)

		# log before-action scene information
		old_scene_info, old_po_ors = self._get_scene_info_before()
		output.update(old_scene_info)

		# only recognition data or not
		if not only_recognition:

			# sample action
			policy = self._action_sampler(output)
			if policy is not None:
				x_pixel, y_pixel, z_pixel, x_coord, y_coord, z_coord, direction_idx, direction_angle = policy
			else:
				return None

			# log action by pixel and coordinates
			output['action_pixel'] = np.array([direction_idx, y_pixel, x_pixel, z_pixel])
			output['action_coord'] = np.array([direction_angle, x_coord, y_coord, z_coord])

			# log after-action scene information
			new_scene_info = self._get_scene_info_after(old_po_ors)
			output.update(new_scene_info)

		return output

	def _random_drop(
			self, 
			object_types, 
			knowledge, 
			num_objects, 
			enable_stacking
		):

		# enable stacking
		self.enable_stacking = enable_stacking

		# distance threshold
		distance_threshold = 0.07 if not enable_stacking else 0.05
		# distance_threshold = 0.02 if not enable_stacking else 0.05

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
					[[0, 1, 2], [1, 2, 0], [2, 0, 1]]
				)
				size_box = size_box[exchange_dims]
				size_x = int(np.round(size_box[0]))
				size_y = int(np.round(size_box[1]))
				size_z = int(np.round(size_box[2]))

				# object pose
				if not enable_stacking:
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

	def _create_box(self, size_x, size_y, size_z, position, orientation, scale_box=500, enable_stacking=False):
				
		# voxelize object
		md = np.ones([size_x, size_y, size_z])
		coord = (np.asarray(np.nonzero(md)).T + 0.5 - np.array([size_x/2, size_y/2, size_z/2]))

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
			p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation
		)
		p.changeDynamics(
			body_id, 
			-1, 
			# spinningFriction=0.002 if not enable_stacking else 0.2, 
			# lateralFriction=0.4 if not enable_stacking else 0.6,
			spinningFriction=0.2,
			lateralFriction=0.6,
			rollingFriction=0.1,
			mass=0.737*size_x*size_y*size_z/(125*1000)
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=np.concatenate([1 * np.random.rand(3), [1]]))
		self.object_ids.append(body_id)
		self.voxel_coord[body_id] = coord / scale_box

		# open3d mesh
		mesh_box = o3d.geometry.TriangleMesh.create_box(
			width = size_x/scale_box, 
			height = size_y/scale_box, 
			depth = size_z/scale_box
		)
		mesh_box.translate([-size_x/(2*scale_box), -size_y/(2*scale_box), -size_z/(2*scale_box)]) # match center to the origin
		self.meshes[body_id] = mesh_box
		object_info = {
			'type': 'box', 
			'size': [size_x/scale_box, size_y/scale_box, size_z/scale_box]
		}
		self.object_info.append(object_info)

		return body_id

	def _create_cylinder(self, size_r, size_h, position, orientation, scale_cylinder=500):
				
		# voxelize object
		X = [r * np.cos(np.linspace(0, 2*np.pi, num=8*r, endpoint=False)) if r != 0 else [0] for r in range(size_r+1)]
		Y = [r * np.sin(np.linspace(0, 2*np.pi, num=8*r, endpoint=False)) if r != 0 else [0] for r in range(size_r+1)]
		X = np.expand_dims(np.concatenate(X), axis=1)
		Y = np.expand_dims(np.concatenate(Y), axis=1)
		Z = np.expand_dims(np.repeat(np.arange(size_h)+0.5-size_h/2, len(X)), 1)
		X = np.tile(X, (size_h, 1))
		Y = np.tile(Y, (size_h, 1))
		coord = np.concatenate((X, Y, Z), axis=1)

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
			p.getQuaternionFromEuler(orientation) if len(orientation) == 3 else orientation
		)
		p.changeDynamics(
			body_id, 
			-1, 
			# spinningFriction=0.002, 
			# lateralFriction=0.4,
			spinningFriction=0.2,
			lateralFriction=0.6,
			rollingFriction=0.1,
			mass=0.585*np.pi*size_r*size_r*size_h/(125*1000)
		)
		p.changeVisualShape(
			body_id, 
			-1, 
			rgbaColor=np.concatenate([1 * np.random.rand(3), [1]])
		)
		self.object_ids.append(body_id)
		self.voxel_coord[body_id] = coord / scale_cylinder
		
		# open3d mesh
		mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=size_r/scale_cylinder, height=size_h/scale_cylinder, resolution=100, split=10)
		self.meshes[body_id] = mesh_cylinder
		object_info = {'type': 'cylinder', 'size': [size_r/scale_cylinder, size_r/scale_cylinder, size_h/scale_cylinder]}
		self.object_info.append(object_info)

		return body_id

	def _check_workspace(self):
		for obj_id in self.object_ids:
			position, orientation = p.getBasePositionAndOrientation(obj_id)
			coord = self._get_coord(obj_id, position, orientation)

			larger_than_x_min = (coord[:, 0] >= self.workspace_bounds[0, 0])
			smaller_than_x_max = (coord[:, 0] <= self.workspace_bounds[0, 1])
			larger_than_y_min = (coord[:, 1] >= self.workspace_bounds[1, 0])
			smaller_than_y_max = (coord[:, 1] <= self.workspace_bounds[1, 1])
			larger_than_z_min = (coord[:, 2] >= self.workspace_bounds[2, 0])
			smaller_than_z_max = (coord[:, 2] <= self.workspace_bounds[2, 1])

			valid_idxs = np.logical_and(
				np.logical_and(
					np.logical_and(larger_than_x_min, smaller_than_x_max),
					np.logical_and(larger_than_y_min, smaller_than_y_max)
				),
				np.logical_and(larger_than_z_min, smaller_than_z_max)
			).all()

			if not valid_idxs:
				# print(f'larger_than_x_min: {larger_than_x_min.all()}')
				# print(f'smaller_than_x_max: {smaller_than_x_max.all()}')
				# print(f'larger_than_y_min: {larger_than_y_min.all()}')
				# print(f'smaller_than_y_max: {smaller_than_y_max.all()}')
				# print(f'larger_than_z_min: {larger_than_z_min.all()}')
				# print(f'smaller_than_z_max: {smaller_than_z_max.all()}')
				return False

		return True

	def _check_stacked(self):
		po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]
		mask_3d = self._get_mask_scene_flow_3d(po_ors, get_scene_flow=False)

		# bottom and top 2d mask
		mask_2d_bot = mask_3d[:, :, 0]
		mask_2d_top = np.max(mask_3d, axis=2)

		# count pixels
		pixel_cnt_bot = np.zeros(len(self.object_ids) + 1)
		uniques, counts = np.unique(mask_2d_bot, return_counts=True)
		pixel_cnt_bot[uniques] = counts

		pixel_cnt_top = np.zeros(len(self.object_ids) + 1)
		uniques, counts = np.unique(mask_2d_top, return_counts=True)
		pixel_cnt_top[uniques] = counts

		# if the bottom 2d mask and the top 2d mask have different pixel counts, it means stacked
		if (np.abs(pixel_cnt_bot - pixel_cnt_top) < 10).all():
			return False
		else:
			return True

	def _check_occlusion(self):
		po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]
		mask, image_size, point_id, x, y = self._get_mask_scene_flow_2d(po_ors, get_scene_flow=False, occlusion=True)

		obj_num = len(self.object_ids)
		mask_sep = np.zeros([obj_num + 1, image_size[0], image_size[1]])
		mask_sep[point_id, x, y] = 1

		for i in range(obj_num):
			tot_pixel_num = np.sum(mask_sep[i + 1])
			vis_pixel_num = np.sum((mask == (i+1)).astype(np.float))

			if vis_pixel_num < 0.4 * tot_pixel_num:
				return True

		return False

	def _calculate_visibility(self):
	
		# load position and orientation
		po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]

		# initialize
		coords = []
		point_id = []
		cur_cnt = 0

		# camera information
		camera_view_matrix = np.array(self.sim.camera_params[0]['camera_view_matrix']).reshape(4, 4).T
		camera_intr = self.sim.camera_params[0]['camera_intr']
		image_size = self.sim.camera_params[0]['camera_image_size']

		# mask image initialize
		obj_num = len(self.object_ids)
		mask = np.zeros([image_size[0], image_size[1]])
		mask_full = np.zeros([obj_num + 1, image_size[0], image_size[1]])

		# voxelize each object
		for obj_id, po_or in zip(self.object_ids, po_ors):
			cur_cnt += 1

			position, orientation = po_or
			coord = self._get_coord(obj_id, position, orientation)

			coords.append(coord)            
			point_id.append([cur_cnt for _ in range(coord.shape[0])])

		# project to depth image
		point_id = np.concatenate(point_id)
		coords_world = np.concatenate(coords)
		coords_2d = project_pts_to_2d(
			coords_world.T, 
			camera_view_matrix, 
			camera_intr
		)
		y = np.round(coords_2d[0]).astype(int)
		x = np.round(coords_2d[1]).astype(int)
		depth = coords_2d[2]

		# extract valid indices
		valid_idx = np.logical_and(
			np.logical_and(x >= 0, x < image_size[0]),
			np.logical_and(y >= 0, y < image_size[1])
		)
		x = x[valid_idx]
		y = y[valid_idx]
		depth = depth[valid_idx]
		point_id = point_id[valid_idx]

		# calculate mask image
		sort_id = np.argsort(-depth)
		x = x[sort_id]
		y = y[sort_id]
		point_id = point_id[sort_id]

		# mask image
		mask[x, y] = point_id
		mask_full[point_id, x, y] = 1

		# calculate visable ratio
		vis_ratios = []
		for i in range(obj_num):
			tot_pixel_num = np.sum(mask_full[i + 1])
			vis_pixel_num = np.sum((mask == (i+1)).astype(float))
			vis_ratio = vis_pixel_num / tot_pixel_num
			vis_ratios.append(vis_ratio)
		vis_ratios = np.array(vis_ratios)

		return vis_ratios

	def _get_scene_info_before(self):

		# before-action positions and orientations
		old_po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]

		positions_old = np.array([old_po_or[0] for old_po_or in old_po_ors])
		orientations_old = np.array([old_po_or[1] for old_po_or in old_po_ors])

		self._get_image()

		self._move_meshes(old_po_ors)   # meshes are used to augment mask and generate gt point cloud

		# point cloud
		pc_data = self._get_pc(
			self.color_image_large, 
			self.depth_image_large, 
			self.mask_image_large
		)

		# ground truth point cloud
		pc_gt = self._get_gt_pc()

		# # tsdf
		# tsdf = get_volume(
		# 	self.color_image_large, 
		# 	self.depth_image_large,
		# 	self.sim.camera_params[0]['camera_intr'], 
		# 	self.sim.camera_params[0]['camera_pose'], 
		# 	deepcopy(self.workspace_bounds), 
		# 	self.voxel_size
		# )

		# # 3d masks
		# mask_3d = self._get_mask_scene_flow_3d(
		# 	old_po_ors, 
		# 	get_scene_flow=False
		# )

		# # organized point cloud
		# organized_pc, _ = project_pts_to_3d(
		# 	self.color_image_small, 
		# 	self.depth_image_small,
		# 	self.sim.camera_params[0]['camera_intr'], 
		# 	self.sim.camera_params[0]['camera_pose']
		# )

		# update scene info
		scene_info = {
			'positions_old': positions_old,
			'orientations_old': orientations_old,
			'pc_gt': pc_gt,
			# 'tsdf': tsdf,
			# 'mask_3d_old': mask_3d,
			'mask_2d_old': self.mask_image_small,
			# 'organized_pc': organized_pc,
			'depth_image_old': self.depth_image_small,
			'color_image_old': self.color_image_small,
		}
		scene_info.update(pc_data)

		return scene_info, old_po_ors

	def _get_scene_info_after(self, old_po_ors):
		
		# after-action positions and orientations
		new_po_ors = [p.getBasePositionAndOrientation(obj_id) for obj_id in self.object_ids]

		positions_new = np.array([new_po_or[0] for new_po_or in new_po_ors])
		orientations_new = np.array([new_po_or[1] for new_po_or in new_po_ors])

		# 3d, 2d scene_flows
		_, scene_flow_3d = self._get_mask_scene_flow_3d(old_po_ors)
		_, scene_flow_2d = self._get_mask_scene_flow_2d(old_po_ors)

		# 3d mask
		mask_3d = self._get_mask_scene_flow_3d(
			new_po_ors, 
			get_scene_flow=False
		)

		self._get_image()

		scene_info = {
			'positions_new': positions_new,
			'orientations_new': orientations_new,
			'scene_flow_3d': scene_flow_3d,
			'scene_flow_2d': scene_flow_2d,
			'mask_3d_new': mask_3d,
			'mask_2d_new': self.mask_image_small,
			'depth_image_new': self.depth_image_small,
			'color_image_new': self.color_image_small,
		}

		return scene_info

	def _get_image(self):
		self.color_image_large, self.depth_image_large, mask_image_large = self.sim.get_camera_data(self.sim.camera_params[0])
		self.color_image_small, self.depth_image_small, mask_image_small = self.sim.get_camera_data(self.sim.camera_params[0])
		
		self.mask_image_large = np.zeros_like(mask_image_large)
		for i, object_id in enumerate(self.object_ids):
			self.mask_image_large += (mask_image_large == object_id) * (i + 1)

		self.mask_image_small = np.zeros_like(mask_image_small)
		for i, object_id in enumerate(self.object_ids):
			self.mask_image_small += (mask_image_small == object_id) * (i + 1)

	def _move_meshes(self, new_po_ors):
		if self.old_po_ors_for_transform is None:
			for obj_id, po_or in zip(self.object_ids, new_po_ors):
				position, orientation = po_or

				# get mesh
				mesh = self.meshes[obj_id]

				# get T matrix
				R = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3)
				T = get_SE3s(R, np.array(position))

				# tranform mesh
				mesh.transform(T)

		else:
			for obj_id, new_po_or, old_po_or in zip(self.object_ids, new_po_ors, self.old_po_ors_for_transform):
				position_new, orientation_new = new_po_or
				position_old, orientation_old = old_po_or

				# get mesh
				mesh = self.meshes[obj_id]

				# get T_new matrix
				R_new = np.asarray(p.getMatrixFromQuaternion(orientation_new)).reshape(3, 3)
				T_new = get_SE3s(R_new, np.array(position_new))

				# get T_old matrix
				R_old = np.asarray(p.getMatrixFromQuaternion(orientation_old)).reshape(3, 3)
				T_old = get_SE3s(R_old, np.asarray(position_old))

				# transform mesh
				mesh.transform(np.linalg.inv(T_old))
				mesh.transform(T_new)

		self.old_po_ors_for_transform = new_po_ors

	def _get_coord(self, obj_id, position, orientation, vol_bnds=None, voxel_size=None):
		# if vol_bnds is not None, return coord in voxel, else, return world coord
		coord = self.voxel_coord[obj_id]
		mat = euler2rotm(p.getEulerFromQuaternion(orientation))
		coord = (mat @ (coord.T)).T + np.asarray(position)
		if vol_bnds is not None:
			coord = np.round((coord - vol_bnds[:, 0]) / voxel_size).astype(np.int)
		return coord

	def _get_mask_scene_flow_3d(self, old_po_ors, get_scene_flow=True):
		vol_bnds = self.workspace_bounds
		mask = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds], dtype=np.int)

		if get_scene_flow:
			scene_flow = np.zeros([int((x[1] - x[0] + 1e-7) / self.voxel_size) for x in vol_bnds] + [3])

		cur_cnt = 0
		for obj_id, old_po_or in zip(self.object_ids, old_po_ors):
			cur_cnt += 1

			position, orientation = old_po_or
			old_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

			# valid_idx = np.logical_and(
			# 	np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < mask.shape[0]),
			# 	np.logical_and(
			# 		np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < mask.shape[1]),
			# 		np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < mask.shape[2])
			# 	)
			# )
			# x = old_coord[valid_idx, 1]
			# y = old_coord[valid_idx, 0]
			# z = old_coord[valid_idx, 2]
			valid_idx = np.logical_and(
				np.logical_and(old_coord[:, 0] >= 0, old_coord[:, 0] < mask.shape[0]),
				np.logical_and(
					np.logical_and(old_coord[:, 1] >= 0, old_coord[:, 1] < mask.shape[1]),
					np.logical_and(old_coord[:, 2] >= 0, old_coord[:, 2] < mask.shape[2])
				)
			)
			x = old_coord[valid_idx, 0]
			y = old_coord[valid_idx, 1]
			z = old_coord[valid_idx, 2]

			mask[x, y, z] = cur_cnt

			if get_scene_flow:
				position, orientation = p.getBasePositionAndOrientation(obj_id)
				new_coord = self._get_coord(obj_id, position, orientation, vol_bnds, self.voxel_size)

				motion = new_coord - old_coord

				motion = motion[valid_idx]
				# motion = np.stack([motion[:, 1], motion[:, 0], motion[:, 2]], axis=1)
				motion = np.stack([motion[:, 0], motion[:, 1], motion[:, 2]], axis=1)

				scene_flow[x, y, z] = motion

		if get_scene_flow:
			return mask, scene_flow
		else:
			return mask

	def _get_mask_scene_flow_2d(self, old_po_ors, get_scene_flow=True, occlusion=False):
		old_coords = []
		point_id = []
		cur_cnt = 0

		camera_view_matrix = np.array(self.sim.camera_params[0]['camera_view_matrix']).reshape(4, 4).T
		camera_intr = self.sim.camera_params[0]['camera_intr']
		image_size = self.sim.camera_params[0]['camera_image_size']

		mask = np.zeros([image_size[0], image_size[1]])
		
		for obj_id, po_or in zip(self.object_ids, old_po_ors):
			cur_cnt += 1

			position, orientation = po_or
			old_coord = self._get_coord(obj_id, position, orientation)

			old_coords.append(old_coord)            
			point_id.append([cur_cnt for _ in range(old_coord.shape[0])])

		point_id = np.concatenate(point_id)
		old_coords_world = np.concatenate(old_coords)

		old_coords_2d = project_pts_to_2d(old_coords_world.T, camera_view_matrix, camera_intr)

		y = np.round(old_coords_2d[0]).astype(np.int)
		x = np.round(old_coords_2d[1]).astype(np.int)
		depth = old_coords_2d[2]

		valid_idx = np.logical_and(
			np.logical_and(x >= 0, x < image_size[0]),
			np.logical_and(y >= 0, y < image_size[1])
		)
		x = x[valid_idx]
		y = y[valid_idx]
		depth = depth[valid_idx]
		point_id = point_id[valid_idx]

		sort_id = np.argsort(-depth)
		x = x[sort_id]
		y = y[sort_id]
		point_id = point_id[sort_id]

		mask[x, y] = point_id

		if get_scene_flow:
			new_coords = []

			scene_flow = np.zeros([image_size[0], image_size[1], 3])

			for obj_id in self.object_ids:
				position, orientation = p.getBasePositionAndOrientation(obj_id)
				new_coord = self._get_coord(obj_id, position, orientation)
				new_coords.append(new_coord)

			new_coords_world = np.concatenate(new_coords)

			motion = (new_coords_world - old_coords_world)[valid_idx]

			motion = motion[sort_id]
			motion = np.stack([motion[:, 0], motion[:, 1], motion[:, 2]], axis=1)

			scene_flow[x, y] = motion

		if occlusion:
			return mask, image_size, point_id, x, y
		else:
			if get_scene_flow:
				return mask, scene_flow
			else:
				return mask

	def _get_pc(self, color_image, depth_image, mask_image):
		camera_pose = self.sim.camera_params[0]['camera_pose']
		camera_intr = self.sim.camera_params[0]['camera_intr']

		organized_pc, organized_rgb_pc = project_pts_to_3d(color_image, depth_image, camera_intr, camera_pose)

		pc, rgb_pc, labels = self._get_workspace_pc(organized_pc, organized_rgb_pc, mask_image)

		wo_plane_idxs = pc[:, 2] >= self.workspace_bounds[2, 0] + 0.002
		wo_plane_idxs = np.where(wo_plane_idxs==1)[0].tolist()

		if pc.shape[0] > self.num_pts_down:
			down_idxs = random.sample(range(pc.shape[0]), self.num_pts_down)
		else:
			down_idxs = list(range(pc.shape[0]))
		pc_down = pc[down_idxs]
		rgb_pc_down = rgb_pc[down_idxs]
		labels_down = labels[down_idxs]

		if len(wo_plane_idxs) > self.num_pts_down_wo_plane:
			down_wo_plane_idxs = random.sample(wo_plane_idxs, self.num_pts_down_wo_plane)
			pc_down_wo_plane = pc[down_wo_plane_idxs]
			rgb_pc_down_wo_plane = rgb_pc[down_wo_plane_idxs]
			labels_down_wo_plane = labels[down_wo_plane_idxs]
		else:
			pc_wo_plane = pc[wo_plane_idxs]
			rgb_pc_wo_plane = rgb_pc[wo_plane_idxs]
			labels_wo_plane = labels[wo_plane_idxs]
			pc_down_wo_plane, rgb_pc_down_wo_plane, labels_down_wo_plane = \
				upsample_pointcloud(self.num_pts_down_wo_plane, pc_wo_plane, rgb_pc_wo_plane, labels_wo_plane)

		pc_data = {
			'pc': pc,
			'rgb_pc': rgb_pc,
			'labels': labels,
			'pc_down': pc_down,
			'rgb_pc_down': rgb_pc_down,
			'labels_down': labels_down,
			'pc_down_wo_plane': pc_down_wo_plane,
			'rgb_pc_down_wo_plane': rgb_pc_down_wo_plane,
			'labels_down_wo_plane': labels_down_wo_plane,
		}

		return pc_data

	def _get_workspace_pc(self, organized_pc, rgb_pc, labels):
		pc = organized_pc.reshape(-1, organized_pc.shape[2])
		rgb_pc = rgb_pc.reshape(-1, rgb_pc.shape[2])
		labels = labels.reshape(-1)

		valid_idxs = np.logical_and(
			np.logical_and(
				np.logical_and(pc[:, 0]>=self.workspace_bounds[0, 0], pc[:, 0]<=self.workspace_bounds[0, 1]),
				np.logical_and(pc[:, 1]>=self.workspace_bounds[1, 0], pc[:, 1]<=self.workspace_bounds[1, 1])
			),
			np.logical_and(pc[:, 2]>=self.workspace_bounds[2, 0]-0.001, pc[:, 2]<=self.workspace_bounds[2, 1])
		)

		pc = pc[valid_idxs]
		rgb_pc = rgb_pc[valid_idxs]
		labels = labels[valid_idxs]

		return pc, rgb_pc, labels

	def _get_gt_pc(self):
		gt_pc_list = []
		for obj_id in self.object_ids:
			# get mesh
			mesh = self.meshes[obj_id]
			mesh.compute_vertex_normals()

			# uniform sampling
			pcd = mesh.sample_points_uniformly(number_of_points=self.num_pts_gt)
			gt_pc = np.asarray(pcd.points)
			gt_normals = np.asarray(pcd.normals)
			gt_pc = np.concatenate((gt_pc, gt_normals), axis=1)

			gt_pc_list.append(gt_pc)

		return np.array(gt_pc_list)

	def _action_sampler(self, output):
		
		# parameters
		collision_epsilon = 5e-4
		falldown_epsilon = 5e-4

		# direction candidates
		pushing_directions = np.array([[0., -1., 0.], [0., 1., 0.]])

		# object configs
		object_pcs = output['pc_gt'][:, :, :3]
		object_positions = output['positions_old']
		object_orientations = output['orientations_old']
		object_SO3s = quats_to_matrices(object_orientations)
		object_poses = get_SE3s(object_SO3s, object_positions)

		# pushing samping
		pushing_poses = push_planner(
			object_pcs,
			self.sim.gripper_open_pc,
			object_poses,
			gripper_orientation=self.sim.gripper_orientation,
			pushing_directions=pushing_directions,
			gripper_height=0.01,
			gripper_depth=-0.035,
			spacing_width=0.02,
			data_type='numpy'
		)

		# available (non-collide) poses


		# sample
		object_indices = np.random.permutation(pushing_poses.shape[0])
		for obj_idx in object_indices:
			pushing_dir_indices = np.random.permutation(pushing_poses.shape[1])
			for push_idx in pushing_dir_indices:
					
				# pushing pose
				pushing_pose = pushing_poses[obj_idx, push_idx]
				pushing_direction = pushing_directions[push_idx]
				
				# pushing distance
				pushing_distance = np.random.choice([0.09, 0.09, 0.09])

				# target pose
				target_pose = deepcopy(pushing_pose)
				target_pose[:3, 3] += pushing_direction * pushing_distance

				# pushing
				self.sim.pushing(pushing_pose, target_pose)

				# get previous positions and orientations
				old_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				old_ors = np.array([p.getBasePositionAndOrientation(object_id)[1] for object_id in self.object_ids])

				# go to initial pose for checking collision
				ik_solved = self.sim.down_action(position=push_initial, rotation_angle=direction_angle)

				# reset if robot cannot make the configuration
				if not ik_solved:
					self.sim.robot_go_home()
					self._reset_objects(old_pos, old_ors)
					continue

				# reset if collision is detected
				new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				position_diff = np.linalg.norm(new_pos - old_pos, axis=1).max()
				if position_diff > collision_epsilon:
					self.sim.up_action(position=push_initial, rotation_angle=direction_angle)
					self._reset_objects(old_pos, old_ors)
					continue

				# take action
				ik_solved = self.sim.push_action(position=push_initial, rotation_angle=direction_angle, speed=0.05, distance=0.10)
				self.sim.robot_go_home()

				# reset if robot cannot make the configuration
				if not ik_solved:
					self._reset_objects(old_pos, old_ors)
					continue

				# reset if object falls down
				new_pos = np.array([p.getBasePositionAndOrientation(object_id)[0] for object_id in self.object_ids])
				orientation_diff = np.abs(new_pos[:, 2] - old_pos[:, 2]).max()
				if orientation_diff > falldown_epsilon:
					self._reset_objects(old_pos, old_ors)
					continue

				# reset if any object goes outside workspace
				if not self._check_workspace():
					self._reset_objects(old_pos, old_ors)
					continue

				# convert coordinate to pixel
				x_pixel, y_pixel, z_pixel = self._coord2pixel(x_coord, y_coord, z_coord)

				return x_pixel, y_pixel, z_pixel, x_coord, y_coord, z_coord, direction_idx, direction_angle

			self.sim.robot_go_home()
			
			return None

	def _reset_objects(self, old_pos, old_ors):
		for idx, object_id in enumerate(self.object_ids):
			p.resetBasePositionAndOrientation(object_id, old_pos[idx], old_ors[idx])

	def _coord2pixel(self, x_coord, y_coord, z_coord):
		x_pixel = int((x_coord - self.workspace_bounds[0, 0]) / self.voxel_size)
		y_pixel = int((y_coord - self.workspace_bounds[1, 0]) / self.voxel_size)
		z_pixel = int((z_coord - self.workspace_bounds[2, 0]) / self.voxel_size)
		return x_pixel, y_pixel, z_pixel
	
if __name__ == '__main__':
	env = SimulationShelfEnv(enable_gui=False)
	env.reset(4, ['box', 'cylinder'])

	# if you just want to get the information of the scene, use env._get_scene_info
	output = env._get_scene_info_before()
	print(output.keys())

	# if use the pushing. env.poke() will also give you everything, together with scene flow
	output = env.poke()
	print(output.keys())
