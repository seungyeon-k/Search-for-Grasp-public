import math
import threading
import time
import copy
import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
from copy import deepcopy
from functions.utils import get_SE3s
from control.gripper import Gripper
from functions.lie import exp_se3, matrices_to_quats, log_SO3, exp_so3
from pybullet_blender.pyBulletSimRecorder import PyBulletRecorder

class PybulletShelfSim:
	def __init__(self, enable_gui, blender_recorder=False):
		
		# environment settings
		self.plane_z = -0.8
		self.low_table_position = [0.26, 0, -0.075/2-0.012]
		self.high_shelf_position = [0.61, 0.1, 0.003]
		self.high_shelf_orientation =  [0.0, 0.0, 1., 0.]

		# load gripper and gripper point cloud
		gripper_open = Gripper(np.eye(4), 0.08)
		self.gripper_open_pc = gripper_open.get_gripper_afterimage_pc(
			pc_dtype='numpy'
		)

		# gripper tilt for pushing
		tilt = np.pi/6
		self.gripper_orientation = np.array(
			[[0., 0., 1.], 
			[0., -1., 0.], 
			[1., 0., 0.]]
		) @ np.array(
			[[np.cos(tilt), 0, -np.sin(tilt)],
			[0, 1, 0],
			[np.sin(tilt), 0, np.cos(tilt)]]
		)

		# set workspace bounds
		offset_x_front = -0.009-0.012
		offset_x_back = -0.009
		offset_y = 0.004
		self.workspace_bounds = np.array(
			[
				[self.high_shelf_position[0] - 0.374/2 + offset_x_front, self.high_shelf_position[0] + 0.374/2 + offset_x_back],
				[self.high_shelf_position[1] - 0.767/2 + offset_y, self.high_shelf_position[1] + 0.767/2 - offset_y],
				[self.high_shelf_position[2] + 0.3905 + 0.022/2, self.high_shelf_position[2] + 0.3905 + 0.022/2 + 0.300]
			]
		)
		self.spawn_bounds = np.array(
			[
				[self.high_shelf_position[0] - 0.160, self.high_shelf_position[0] + 0.120],
				[self.high_shelf_position[1] - 0.280, self.high_shelf_position[1] + 0.280],
				[self.high_shelf_position[2] + 0.3905 + 0.022/2, self.high_shelf_position[2] + 0.3905 + 0.022/2 + 0.200]
			]
		)
		self.shelf_bounds = np.array(
			[
				[self.high_shelf_position[0] - 0.767/2, self.high_shelf_position[0] + 0.767/2],
				[self.high_shelf_position[1] + 0.009 - 0.374/2, self.high_shelf_position[1] + 0.009 + 0.374/2],
				[self.high_shelf_position[2] + 0.022/2 + 0.3905, self.high_shelf_position[2] + 0.022 + 0.76]
			]
		)

		# Start blender recorder
		self.blender_recorder = blender_recorder
		if self.blender_recorder:
			self.recorder = PyBulletRecorder()
			self.recorder_time_interval = 5
		else:
			self.recorder = None

		# Start PyBullet simulation
		if enable_gui:
			self._physics_client = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
		else:
			self._physics_client = p.connect(p.DIRECT)  # non-graphical version
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.setGravity(0, 0, -9.8)
		if not self.blender_recorder:
			self.start_simulation_thread()

		# Add ground plane
		self._plane_id = p.loadURDF("plane.urdf", [0, 0, self.plane_z])

		# Add table
		table_path = 'assets/table/low_table_for_blender.urdf'
		self._low_table_id = p.loadURDF(table_path, self.low_table_position, useFixedBase=True)
		if self.blender_recorder:
			self.recorder.register_object(self._low_table_id, table_path)

		# Add shelf
		shelf_path = 'assets/shelf/shelf_for_blender.urdf'
		self._high_shelf_id = p.loadURDF(shelf_path, self.high_shelf_position, self.high_shelf_orientation, useFixedBase=True)
		if self.blender_recorder:
			self.recorder.register_object(self._high_shelf_id, shelf_path)

		# Add Franka Panda Emika robot
		if self.blender_recorder:
			robot_path = 'assets/panda/panda_with_gripper_full.urdf'
		else:
			robot_path = 'assets/panda/panda_with_gripper.urdf'
		# robot_path = 'assets/panda_dae/panda_with_gripper.urdf'
		self._robot_body_id = p.loadURDF(robot_path, [0.0, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
		if self.blender_recorder:
			self.recorder.register_object(self._robot_body_id, robot_path)

		# Get revolute joint indices of robot (skip fixed joints)
		robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
		self._robot_joint_indices = [x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._robot_joint_lower_limit = [x[8] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._robot_joint_upper_limit = [x[9] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
		self._finger_joint_indices = [8, 9]
		self._joint_epsilon = 0.01  # joint position threshold in radians for blocking calls (i.e. move until joint difference < epsilon)

		# Move robot to home joint configuration
		self._robot_home_joint_config = [0.0301173169862714, -1.4702106391932968, 0.027855688427362513, -2.437557753144649, 0.14663284881434122, 2.308719465520647, 0.7012385825324389]
		self.robot_go_home()

		# robot end-effector index
		self._robot_EE_joint_idx = 7
		self._robot_tool_joint_idx = 9
		self._robot_tool_tip_joint_idx = 9

		# Set friction coefficients for gripper fingers
		p.changeDynamics(
			self._robot_body_id, 7,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)
		p.changeDynamics(
			self._robot_body_id, 8,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)
		p.changeDynamics(
			self._robot_body_id, 9,
			lateralFriction=1, # 0.1
			spinningFriction=1, # 0.1
			rollingFriction=1,
			frictionAnchor=True
		)

		# get out camera view matrix
		self.ee_state = p.getLinkState(self._robot_body_id, 7)
		self.ee_rot = np.asarray(p.getMatrixFromQuaternion(self.ee_state[5])).reshape(3,3)
		self.ee_pose = get_SE3s(self.ee_rot, np.array(self.ee_state[4]))

		# camera pose (rgb frame)
		# self.kinect_pose = np.array([[0., 0., 1., self.high_shelf_position[0] - 0.75],
		# 							 [-1., 0., 0., 0.],
		# 							 [0., -1. ,0., 0.71016075 + self.high_shelf_position[2]],
		# 							 [0, 0, 0, 1]])
		self.kinect_pose = np.array([
			[-0.43471579, -0.10620786,  0.894283,   -0.04457319],
 			[-0.9005385 ,  0.05926315, -0.43071834,  0.44833383],
 			[-0.00725236, -0.99257633, -0.12140689,  0.65361773],
 			[ 0.        ,  0.        ,  0.        ,  1.        ]
		])
		self.kinect_intrinsic = np.array(
			[609.7949829101562, 609.4755859375, 640.93017578125, 368.19635009765625]
		) # fx, fy, px, py

		# camera list
		self.camera_params = {
			# azure kinect
			0: self._get_camera_param(
				camera_pose = self.kinect_pose,
				camera_intrinsic = self.kinect_intrinsic,
				camera_image_size=[720, 1280]
			),
		}

	#############################################################
	#################### SIMULATION STEP ########################
	#############################################################

	# start simulation 
	def start_simulation_thread(self):
		step_sim_thread = threading.Thread(target=self.step_simulation)
		step_sim_thread.daemon = True
		step_sim_thread.start()

	# Step through simulation time
	def step_simulation(self):
		interval_stamp = 0
		while True:
			p.stepSimulation()
			time.sleep(0.0001)
			if self.blender_recorder:
				interval_stamp += 1
				if interval_stamp == self.recorder_time_interval:
					self.recorder.add_keyframe()
					interval_stamp = 0

	#############################################################
	################# BLENDER RECORDER UTILS ####################
	#############################################################

	# record
	def sim_recorder_register_object(self, body_id, urdf_path, global_color=None):
		if self.recorder is None:
			raise ValueError('recorder is not defined in pybullet sim!')
		self.recorder.register_object(body_id, urdf_path, global_color=global_color)

	# save
	def sim_recorder_save(self, save_name):
		if self.recorder is None:
			raise ValueError('recorder is not defined in pybullet sim!')
		self.recorder.save(save_name)		

	#############################################################
	##################### GET PARAMETERS ########################
	#############################################################

	def _get_shelf_info(self):
		
		# thickness
		thickness = 0.2 # side : 0.018, upper : 0.022
  
		# Make open3d box
		shelf_bottom_size = [0.392, 0.803, 0.022]
		shelf_left_size = [0.392, thickness, 0.759]
		shelf_right_size = [0.392, thickness, 0.759]
		shelf_middle_size = [0.374, 0.767, 0.022]
		shelf_back_size = [0.018, 0.767, 0.759]
		shelf_upper_size = [0.392, 0.803, thickness]

		# translates
		shelf_bottom_pos = [0, 0, 0]
		shelf_left_pos = [0, 0.3745 + thickness/2, 0.3905]
		shelf_right_pos = [0, -0.3745 - thickness/2, 0.3905]
		shelf_middle_pos = [0.009, 0, 0.3905]
		shelf_back_pos = [-0.187, 0, 0.3905]
		shelf_upper_pos = [0, 0, 0.759 + thickness/2]

		shelf_position = np.array(self.high_shelf_position)
		shelf_orientation = np.array(self.high_shelf_orientation)

		shelf_info = {
			'shelf_bottom': {
				'size': shelf_bottom_size,
				'position': shelf_bottom_pos
			},
			'shelf_left': {
				'size': shelf_left_size,
				'position': shelf_left_pos
			},
			'shelf_right': {
				'size': shelf_right_size,
				'position': shelf_right_pos
			},
			'shelf_middle': {
				'size': shelf_middle_size,
				'position': shelf_middle_pos
			},
			'shelf_back': {
				'size': shelf_back_size,
				'position': shelf_back_pos
			},
			'shelf_upper': {
				'size': shelf_upper_size,
				'position': shelf_upper_pos
			},
			'global_position': shelf_position,
			'global_orientation': shelf_orientation
		}

		return shelf_info

	# def _get_camera_param(
	# 		self, 
	# 		camera_pose=None, 
	# 		camera_view_matrix=None, 
	# 		camera_position=None, 
	# 		camera_image_size=None
	# 	):

	# 	if camera_pose is None:
	# 		camera_lookat = [0.405, 0, 0]
	# 		camera_up_direction = [0, camera_position[2], -camera_position[1]]
	# 		camera_view_matrix = p.computeViewMatrix(camera_position, camera_lookat, camera_up_direction)
	# 		camera_pose = np.linalg.inv(np.array(camera_view_matrix).reshape(4, 4).T)
	# 		camera_pose[:, 1:3] = -camera_pose[:, 1:3]
	# 	else:
	# 		camera_view_matrix = copy.deepcopy(camera_pose)
	# 		camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
	# 		camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)
			
	# 	camera_z_near = 0.01
	# 	camera_z_far = 20
	# 	camera_fov_w = 75
	# 	camera_focal_length = (float(camera_image_size[1]) / 2) / np.tan((np.pi * camera_fov_w / 180) / 2)
	# 	camera_fov_h = (math.atan((float(camera_image_size[0]) / 2) / camera_focal_length) * 2 / np.pi) * 180
	# 	camera_projection_matrix = p.computeProjectionMatrixFOV(
	# 		fov=camera_fov_h,
	# 		aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
	# 		nearVal=camera_z_near,
	# 		farVal=camera_z_far
	# 	)  # notes: 1) FOV is vertical FOV 2) aspect must be float
	# 	camera_intrinsics = np.array(
	# 		[[camera_focal_length, 0, float(camera_image_size[1]) / 2],
	# 		 [0, camera_focal_length, float(camera_image_size[0]) / 2],
	# 		 [0, 0, 1]])
	# 	camera_param = {
	# 		'camera_image_size': camera_image_size,
	# 		'camera_intr': camera_intrinsics,
	# 		'camera_pose': camera_pose,
	# 		'camera_view_matrix': camera_view_matrix,
	# 		'camera_projection_matrix': camera_projection_matrix,
	# 		'camera_z_near': camera_z_near,
	# 		'camera_z_far': camera_z_far
	# 	}
	# 	return camera_param

	def _get_camera_param(
			self, 
			camera_pose,
			camera_intrinsic,
			camera_image_size
		):

		# modified camera intrinsic
		fx = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
		fy = (camera_intrinsic[0] + camera_intrinsic[1]) / 2
		px = float(camera_image_size[1]) / 2
		py = float(camera_image_size[0]) / 2

		# camera view matrix
		camera_view_matrix = copy.deepcopy(camera_pose)
		camera_view_matrix[:, 1:3] = -camera_view_matrix[:, 1:3]
		camera_view_matrix = np.linalg.inv(camera_view_matrix).T.reshape(-1)
		
		# camera z near/far values (arbitrary value)
		camera_z_near = 0.01
		camera_z_far = 20

		# # camera intrinsic matrix
		# camera_intrinsic_matrix = np.array(
		# 	[[camera_intrinsic[0], 0, camera_intrinsic[2]],
		# 	 [0, camera_intrinsic[1], camera_intrinsic[3]],
		# 	 [0, 0, 1]]
		# )

		# camera intrinsic matrix
		camera_intrinsic_matrix = np.array(
			[[fx, 0, px],
			 [0, fy, py],
			 [0, 0, 1]]
		)

		# camera projection matrix
		camera_fov_h = (math.atan(py / fy) * 2 / np.pi) * 180
		camera_projection_matrix = p.computeProjectionMatrixFOV(
			fov=camera_fov_h,
			aspect=float(camera_image_size[1]) / float(camera_image_size[0]),
			nearVal=camera_z_near,
			farVal=camera_z_far
		)  

		camera_param = {
			'camera_image_size': camera_image_size,
			'camera_intr': camera_intrinsic_matrix,
			'camera_pose': camera_pose,
			'camera_view_matrix': camera_view_matrix,
			'camera_projection_matrix': camera_projection_matrix,
			'camera_z_near': camera_z_near,
			'camera_z_far': camera_z_far
		}
		return camera_param

	# Get latest RGB-D image
	def get_camera_data(self, cam_param):
		camera_data = p.getCameraImage(cam_param['camera_image_size'][1], cam_param['camera_image_size'][0],
									   cam_param['camera_view_matrix'], cam_param['camera_projection_matrix'],
									   shadow=1, renderer=p.ER_TINY_RENDERER)
		color_image = np.asarray(camera_data[2]).reshape(
			[cam_param['camera_image_size'][0], cam_param['camera_image_size'][1], 4]
		)[:, :, :3]  # remove alpha channel
		z_buffer = np.asarray(camera_data[3]).reshape(cam_param['camera_image_size'])
		camera_z_near = cam_param['camera_z_near']
		camera_z_far = cam_param['camera_z_far']
		depth_image = (2.0 * camera_z_near * camera_z_far) / (
			camera_z_far + camera_z_near - (2.0 * z_buffer - 1.0) * (
				camera_z_far - camera_z_near
			)
		)
		mask_image = np.asarray(camera_data[4]).reshape(cam_param['camera_image_size'][0:2])
		return color_image, depth_image, mask_image

	#############################################################
	#################### CONTROLLER - SIMULATION ################
	#############################################################

	# robot initialize
	def robot_go_home(self, blocking=True, speed=0.1):
		self.move_joints(self._robot_home_joint_config, blocking, speed)

	def reset_robot(self):
		for i in self._robot_joint_indices:
			p.resetJointState(self._robot_body_id, i, self._robot_home_joint_config[i])

	# pick and place
	def pick_and_place(
			self, 
			initial_SE3, target_SE3, 
			approach_distance=0.3
		):
		
		# declare speed
		speed_gripper = 0.01
		speed = 0.001
		staright_speed = 0.01
		grasp_force = 50

		# get poses
		position = deepcopy(initial_SE3[:3, 3])
		approach_direction = deepcopy(initial_SE3[:3, 2])
		orientation = matrices_to_quats(initial_SE3[:3, :3])
		position_target = deepcopy(target_SE3[:3, 3])
		approach_direction_target = deepcopy(target_SE3[:3, 2])
		orientation_target = matrices_to_quats(target_SE3[:3, :3])

		# print(approach_direction, approach_direction_target)

		# via-points
		position_init = position - approach_distance * approach_direction
		position_target_init = position_target - approach_distance * approach_direction_target

		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		if not self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True
		):
			return False # before approach
		if not self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		):
			return False # approach
		self.grasp_object(
			force=grasp_force,
			blocking=True,
		) # grasp
		if not self.move_carts_straight(
			position=position_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		):
			return False # back
		if not self.move_carts(
			position=position_target_init,
			orientation=orientation,
			speed=speed,
			blocking=True
		):
			return False # target back
		if not self.move_carts_straight(
			position=position_target,
			orientation=orientation_target,
			speed=staright_speed,
			blocking=True
		):
			return False # target
		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		if not self.move_carts_straight(
			position=position_target_init,
			orientation=orientation_target,
			speed=staright_speed,
			blocking=True
		):
			return False # target init
		self.robot_go_home(speed=speed)

		return True

	# pushing action
	def pushing(
			self, 
			initial_SE3, target_SE3, 
			approach_distance=0.3, 
		):
		
		# declare speed
		speed = 0.01
		staright_speed = 0.01
		speed_gripper = 0.01

		# get poses
		position = deepcopy(initial_SE3[:3, 3])
		approach_direction = deepcopy(initial_SE3[:3, 2])
		orientation = matrices_to_quats(initial_SE3[:3, :3])
		position_target = deepcopy(target_SE3[:3, 3])
		approach_direction_target = deepcopy(target_SE3[:3, 2])

		# via-points
		position_init = position - approach_distance * approach_direction
		position_target_init = position_target - approach_distance * approach_direction_target

		self.move_gripper(
			0.0,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		if not self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True
		):
			return False # before approach
		if not self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		):
			return False # approach
		self.move_carts_straight(
			position=position_target,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		) # push
		if not self.move_carts_straight(
			position=position_target_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		):
			return False # backward
		self.robot_go_home(speed=speed)

		return True
  
	# target retrieval
	def target_retrieval(
			self, 
			target_SE3, 
			approach_distance=0.3, 
		):
		
		# declare speed
		speed = 0.001
		staright_speed = 0.01
		speed_gripper = 0.01
		grasp_force = 100

		# get poses
		position = deepcopy(target_SE3[:3, 3])
		approach_direction = deepcopy(target_SE3[:3, 2])
		orientation = matrices_to_quats(target_SE3[:3, :3])

		# via-points
		position_init = position - approach_distance * approach_direction

		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		if not self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True
		):
			return False # before approach
		if not self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		): 
			return False # approach
		self.grasp_object(
			force=grasp_force,
			blocking=True,
		) # grasp
		if not self.move_carts_straight(
			position=position_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True
		):
			return False # retrieve
		self.robot_go_home(speed=speed)

	#############################################################
	#################### CONTROLLER - REALWORLD #################
	#############################################################

	# pick and place
	def pick_and_place_realworld(
			self, 
			initial_SE3, target_SE3, 
			approach_distance=0.3
		):
		
		# declare speed
		speed_gripper = 0.01
		speed = 0.001
		staright_speed = 0.01
		grasp_force = 50
		speed_real = 0.5
		straight_speed_real = 0.4
		
		# get poses
		position = deepcopy(initial_SE3[:3, 3])
		approach_direction = deepcopy(initial_SE3[:3, 2])
		orientation = matrices_to_quats(initial_SE3[:3, :3])
		position_target = deepcopy(target_SE3[:3, 3])
		approach_direction_target = deepcopy(target_SE3[:3, 2])
		orientation_target = matrices_to_quats(target_SE3[:3, :3])

		# print(approach_direction, approach_direction_target)

		# via-points
		position_init = position - approach_distance * approach_direction
		position_target_init = position_target - approach_distance * approach_direction_target

		# list
		total_joint_angle_list = []
		vel_list = []

		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		total_joint_angle_list += [['move_gripper', 0.04]]
		vel_list += [speed_real]		
		success, joint_angle = self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True,
			return_joint_angle=True
		) # before approach
		if success:
			total_joint_angle_list += [joint_angle]
			vel_list += [speed_real]
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # approach
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		self.grasp_object(
			force=grasp_force,
			blocking=True,
		) # grasp
		total_joint_angle_list += [['grasp', 0.00]]
		vel_list += [speed_real]		
		success, joint_angle_list = self.move_carts_straight(
			position=position_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # back
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		success, joint_angle = self.move_carts(
			position=position_target_init,
			orientation=orientation,
			speed=speed,
			blocking=True,
			return_joint_angle=True
		) # target back
		if success:
			total_joint_angle_list += [joint_angle]
			vel_list += [speed_real]
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position_target,
			orientation=orientation_target,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # target
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		total_joint_angle_list += [['move_gripper', 0.04]]
		vel_list += [speed_real]	
		success, joint_angle_list = self.move_carts_straight(
			position=position_target_init,
			orientation=orientation_target,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # target init
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []

		return True, total_joint_angle_list, vel_list, vel_list

	# pushing action
	def pushing_realworld(
			self, 
			initial_SE3, target_SE3, 
			approach_distance=0.3, 
		):
		
		# declare speed
		speed = 0.01
		staright_speed = 0.01
		speed_gripper = 0.01
		speed_real = 0.5
		straight_speed_real = 0.4

		# get poses
		position = deepcopy(initial_SE3[:3, 3])
		approach_direction = deepcopy(initial_SE3[:3, 2])
		orientation = matrices_to_quats(initial_SE3[:3, :3])
		position_target = deepcopy(target_SE3[:3, 3])
		approach_direction_target = deepcopy(target_SE3[:3, 2])

		# via-points
		position_init = position - approach_distance * approach_direction
		position_target_init = position_target - approach_distance * approach_direction_target

		# list
		total_joint_angle_list = []
		vel_list = []

		self.move_gripper(
			0.0,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		total_joint_angle_list += [['move_gripper', 0.00]]
		vel_list += [speed_real]		
		success, joint_angle = self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True,
			return_joint_angle=True
		) # before approach
		if success:
			total_joint_angle_list += [joint_angle]
			vel_list += [speed_real]
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # approach
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position_target,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # push
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position_target_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # backward
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []

		return True, total_joint_angle_list, vel_list, vel_list
  
	# target retrieval
	def target_retrieval_realworld(
			self, 
			target_SE3, 
			approach_distance=0.3, 
		):
		
		# declare speed
		speed = 0.001
		staright_speed = 0.01
		speed_gripper = 0.01
		grasp_force = 100
		speed_real = 0.5
		straight_speed_real = 0.4
		
		# get poses
		position = deepcopy(target_SE3[:3, 3])
		approach_direction = deepcopy(target_SE3[:3, 2])
		orientation = matrices_to_quats(target_SE3[:3, :3])

		# via-points
		position_init = position - approach_distance * approach_direction

		# list
		total_joint_angle_list = []
		vel_list = []

		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper
		total_joint_angle_list += [['move_gripper', 0.04]]
		vel_list += [speed_real]		
		success, joint_angle = self.move_carts(
			position=position_init,
			orientation=orientation,
			speed=speed,
			blocking=True,
			return_joint_angle=True
		) # before approach
		if success:
			total_joint_angle_list += [joint_angle]
			vel_list += [speed_real]
		else:
			return False, [], [], []
		success, joint_angle_list = self.move_carts_straight(
			position=position,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # retrieve 
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		self.grasp_object(
			force=grasp_force,
			blocking=True,
		) # grasp
		total_joint_angle_list += [['grasp', 0.00]]
		vel_list += [speed_real]		
		success, joint_angle_list = self.move_carts_straight(
			position=position_init,
			orientation=orientation,
			speed=staright_speed,
			blocking=True,
			return_joint_angle=True
		) # retrieve
		if success:
			total_joint_angle_list += joint_angle_list
			vel_list += [straight_speed_real] * len(joint_angle_list)
		else:
			return False, [], [], []
		self.move_gripper(
			0.04,
			speed=speed_gripper,
			blocking=True,
		) # open gripper

		return True, total_joint_angle_list, vel_list, vel_list

	#############################################################
	#################### ACTION PRIMITIVES ######################
	#############################################################

	# move joints
	def move_joints(self, target_joint_state, blocking=False, speed=0.03):
		
		# for i in self._robot_joint_indices:
		# 	p.resetJointState(self._robot_body_id, self._robot_joint_indices[i], target_joint_state[i])
		# time.sleep(1.0)
		
		# move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._robot_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			positionGains=speed * np.array([1, 1, 1, 1, 1, 2, 1])
   		)

		# Block call until joints move to target configuration
		if blocking:
			actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
			timeout_t0 = time.time()
			while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
						   range(6)]):
				if time.time() - timeout_t0 > 3:
					return False
				actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
				time.sleep(0.001)
			return True
		else:
			return 1

	# move in cartesian coordinates
	def move_carts(self, position, orientation, blocking=False, speed=0.03, return_joint_angle=False):
		
		# inverse kinematics
		target_joint_state = np.array(
			p.calculateInverseKinematics(
				self._robot_body_id,  
				self._robot_EE_joint_idx, 
				position, 
				orientation,
				maxNumIterations=10000, 
				residualThreshold=.0001,
				lowerLimits=self._robot_joint_lower_limit,
				upperLimits=self._robot_joint_upper_limit
		))

		# except gripper target state
		target_joint_state = target_joint_state[:len(self._robot_joint_indices)]
	
		# move joints
		if return_joint_angle:
			return self.move_joints(
				target_joint_state, 
				blocking=blocking, 
				speed=speed
			), target_joint_state
		else:
			return self.move_joints(
				target_joint_state, 
				blocking=blocking, 
				speed=speed
			)

	# move in cartesian coordinates with "straight line"
	def move_carts_straight(self, position, orientation, blocking=False, speed=0.03, return_joint_angle=False):
		
		# num of segment
		n_segment = 10

		# define segments of straight line motion
		ee_state = p.getLinkState(
			self._robot_body_id, 
			self._robot_EE_joint_idx
		)
		position_initial = deepcopy(np.array(ee_state[4]))
		# orientation_initial = np.asarray(p.getMatrixFromQuaternion(self.ee_state[5])).reshape(3,3)
		# w_initial = log_SO3(orientation_initial)
		# w_final = log_SO3(np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3,3))
		
		target_joint_state_list = []
		for i in range(n_segment):
			# inverse kinematics
			target_joint_state = np.array(
				p.calculateInverseKinematics(
					self._robot_body_id, 
					self._robot_EE_joint_idx, 
					position_initial + (position - position_initial) * (i + 1) / n_segment, 
					orientation,
					#matrices_to_quats(exp_so3(w_initial + (w_final - w_initial) * (i + 1) / n_segment)),
					maxNumIterations=10000, 
					residualThreshold=.0001,
					lowerLimits=self._robot_joint_lower_limit,
					upperLimits=self._robot_joint_upper_limit
			))

			# except gripper target state
			target_joint_state = target_joint_state[:len(self._robot_joint_indices)]
			if return_joint_angle:
				target_joint_state_list.append(target_joint_state)
			
			# move joints		
			if not self.move_joints(target_joint_state, blocking=blocking, speed=speed):
				if return_joint_angle:
					return False, target_joint_state_list
				else:
					return False

		if blocking:
			if return_joint_angle:
				return True, target_joint_state_list
			else:
				return True
		else:
			return 1

	# move tripper
	def move_gripper(self, target_width, blocking=False, speed=0.03):
		
		# target joint state
		target_joint_state = np.array([target_width, target_width])
		
		# Move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._finger_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			positionGains=speed * np.ones(len(self._finger_joint_indices))
		)

		# Block call until joints move to target configuration
		if blocking:
			actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
			timeout_t0 = time.time()
			while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in range(len(actual_joint_state))]):
				if time.time() - timeout_t0 > 5:
					break
				actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
				time.sleep(0.001)

	# grasp the object with force
	def grasp_object(self, blocking=False, force=200):
		
		# target joint state
		target_joint_state = np.array([0.0, 0.0])
		forces = np.array([force, force])
		
		# Move joints
		p.setJointMotorControlArray(
			self._robot_body_id, 
			self._finger_joint_indices,
			p.POSITION_CONTROL, 
			target_joint_state,
			forces=forces,
			# positionGains=speed * np.ones(len(self._finger_joint_indices))
		)

		# Block call until joints move to target configuration
		if blocking:
			timeout_t0 = time.time()
			while True:
				if time.time() - timeout_t0 > 1:
					break
				time.sleep(0.001)

		