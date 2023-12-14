import numpy as np
import torch
import open3d as o3d
from copy import deepcopy
from functions.lie import get_SE3s, exp_so3

class Gripper:
	def __init__(self, SE3, width=0, collisionBox=False):
		self.hand_SE3 = SE3
		self.gripper_width = width
		if width < 0:
			print("gripper width exceeds minimum width. gripper width is set to 0")
			self.gripper_width = 0
		if width > 0.08:
			print("gripper width exceeds maximum width. gripper width is set to 0.08")
			self.gripper_width = 0.08

		self.hand = o3d.io.read_triangle_mesh("assets/gripper/hand.ply")
		self.hand.compute_vertex_normals()
		self.hand.paint_uniform_color([0.9, 0.9, 0.9])
		self.finger1 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger1.compute_vertex_normals()
		self.finger1.paint_uniform_color([0.7, 0.7, 0.7])
		self.finger2 = o3d.io.read_triangle_mesh("assets/gripper/finger.ply")
		self.finger2.compute_vertex_normals()
		self.finger2.paint_uniform_color([0.7, 0.7, 0.7])

		self.finger1_M = get_SE3s(np.identity(3), np.array([0, self.gripper_width/2, 0.1654/3]))
		self.finger2_M = get_SE3s(exp_so3(np.asarray([0, 0, 1]) * np.pi), np.array([0, -self.gripper_width/2, 0.1654/3]))

		self.finger1_SE3 = np.dot(self.hand_SE3, self.finger1_M)
		self.finger2_SE3 = np.dot(self.hand_SE3, self.finger2_M)
			
		self.hand.transform(self.hand_SE3)
		self.finger1.transform(self.finger1_SE3)
		self.finger2.transform(self.finger2_SE3)
		self.mesh = self.hand + self.finger1 + self.finger2
		self.mesh.compute_vertex_normals()

		# for reachability
		d = 0 # d = 0.008
		self.link6 = o3d.io.read_triangle_mesh("assets/gripper/link6.ply")
		self.link6.compute_vertex_normals()
		self.link6.paint_uniform_color([0.6, 0.1, 0.7])
		self.link6.rotate(
			self.link6.get_rotation_matrix_from_xyz(
				(-np.pi/2, 0, 0)
			),
			center=(0, 0, 0)
		)
		self.link6.translate([-0.088, 0, - 0.107 - d])
		self.link7 = o3d.io.read_triangle_mesh("assets/gripper/link7.ply")
		self.link7.compute_vertex_normals()
		self.link7.paint_uniform_color([0.7, 0.6, 0.1])
		self.link7.rotate(
			self.link7.get_rotation_matrix_from_xyz(
				(0, 0, np.pi / 4)
			),
			center=(0, 0, 0)
		)
		self.link7.translate([0, 0, - 0.107 - d])


	def get_gripper_pc(
			self,
			pc_dtype='numpy',
			number_of_points=2048
		):

		# initialize
		gripper_mesh = self.mesh

		# sample gripper pc
		gripper_pc = gripper_mesh.sample_points_uniformly(
			number_of_points=number_of_points
		)
		gripper_pc.paint_uniform_color([0.5, 0.5, 0.5])

		if pc_dtype == 'numpy':
			return np.asarray(gripper_pc.points)
		elif pc_dtype == 'torch':
			return torch.tensor(np.asarray(gripper_pc.points)).float()

	def get_gripper_with_camera_pc(
			self,
			pc_dtype='numpy',
			number_of_points=2048
		):

		# initialize
		gripper_mesh = self.mesh

		# camera
		camera_size = np.array([0.04, 0.095, 0.125])
		camera = o3d.geometry.TriangleMesh.create_box(width = camera_size[0], height = camera_size[1], depth = camera_size[2])
		camera.translate([0.0625 - camera_size[0]/2, - camera_size[1]/2, 0.0 - camera_size[2]/2])
		camera.compute_vertex_normals()
		gripper_mesh += camera

		# # finger box
		# finger_box_size = np.array([0.02, 0.08, 0.02])
		# finger_box = o3d.geometry.TriangleMesh.create_box(width = finger_box_size[0], height = finger_box_size[1], depth = finger_box_size[2])
		# finger_box.translate([-0.01, 0.007 - camera_size[1]/2, 0.15 - camera_size[2]/2])
		# finger_box.compute_vertex_normals()
		# gripper_mesh += finger_box

		# sample gripper pc
		gripper_pc = gripper_mesh.sample_points_uniformly(
			number_of_points=number_of_points
		)
		gripper_pc.paint_uniform_color([0.5, 0.5, 0.5])

		if pc_dtype == 'numpy':
			return np.asarray(gripper_pc.points)
		elif pc_dtype == 'torch':
			return torch.tensor(np.asarray(gripper_pc.points)).float()
	 
	def get_gripper_afterimage_pc(
			self,
			pc_dtype='numpy',
			number_of_points=2048,
			contain_camera=False
		):

		# afterimage_distance
		distance = 0.2
		n_gripper = 5

		# initialize
		gripper_mesh = deepcopy(self.mesh)
		link6_mesh = deepcopy(self.link6)
		link7_mesh = deepcopy(self.link7)
  
  		# camera
		if contain_camera:
			camera_size = np.array([0.04, 0.095, 0.125])
			camera = o3d.geometry.TriangleMesh.create_box(width = camera_size[0], height = camera_size[1], depth = camera_size[2])
			camera.translate([0.0625 - camera_size[0]/2, - camera_size[1]/2, 0.0 - camera_size[2]/2])
			camera.compute_vertex_normals()
			gripper_mesh += camera

		# append mesh
		for i in range(n_gripper):
			gripper_afterimage_mesh = deepcopy(gripper_mesh)
			link6_afterimage_mesh = deepcopy(link6_mesh)
			link7_afterimage_mesh = deepcopy(link7_mesh)
			if contain_camera:
				camera_afterimage_mesh = deepcopy(camera)
			z_distance = - distance * i / (n_gripper - 1)
			gripper_afterimage_mesh.translate([0, 0, z_distance])
			link6_afterimage_mesh.translate([0, 0, z_distance])
			link7_afterimage_mesh.translate([0, 0, z_distance])
			if contain_camera:
				camera_afterimage_mesh.translate([0, 0, z_distance])
			if i == 0:
				gripper_mesh_total = gripper_afterimage_mesh
			else:
				gripper_mesh_total += gripper_afterimage_mesh
			gripper_mesh_total += link6_afterimage_mesh
			gripper_mesh_total += link7_afterimage_mesh
			if contain_camera:
				gripper_mesh_total += camera_afterimage_mesh

		# sample gripper pc
		gripper_pc_total = gripper_mesh_total.sample_points_uniformly(
			number_of_points=number_of_points
		)
		gripper_pc_total.paint_uniform_color([0.5, 0.5, 0.5])

		# coordinate for debug
		# coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

		# visualize
		# o3d.visualization.draw_geometries([
		# 	gripper_mesh,
		# 	self.link6,
		# 	self.link7, 
		# 	gripper_pc_total, 
		# 	coordinate,
		# ])

		if pc_dtype == 'numpy':
			return np.asarray(gripper_pc_total.points)
		elif pc_dtype == 'torch':
			return torch.tensor(np.asarray(gripper_pc_total.points)).float()