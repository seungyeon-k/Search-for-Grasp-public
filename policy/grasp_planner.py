import torch
from functions.lie_torch import get_SE3s_torch
from copy import deepcopy

def superquadric_grasp_planner(
		SE3, 
		parameters, 
		n_gripper_box=4,
		n_gripper_cyl_r=32,
		n_gripper_cyl_h=4,
		d=0.075, 
		max_width=0.075, 
		ratio=0.8, 
		flip=True,
  		augment_flip=False,
		tilt=True,
		augment_tilt=False,
		desired_dir=torch.tensor([1., 0., 0.]),
		return_bool=False
	):
	"""generate grasp poses for superquadric objects

	Args:
		SE3 (4 x 4 torch tensor): target object's pose
		parameters (5 torch tensor): target object's parameter
		n_gripper (int, optional): the number of grippers per each axis. Defaults to 10.
		d (float, optional): offset for grasp poses. Defaults to 0.08.
		max_width (float, optional): the maximum width that the franka gripper can grasp. Defaults to 0.075.
		ratio (float, optional): ratio for grasp. Defaults to 0.8.
		augment (bool, optional): augment flipped grasp poses. Defaults to False.
		flip (bool, optional): only return flipped grasp poses. Defaults to True.

	Returns:
		gripper_SE3s (n_grasp x 4 x 4 torch tensor): grasp poses
	"""

	# initialize
	a1 = parameters[0]
	a2 = parameters[1]
	a3 = parameters[2]
	e1 = parameters[3]
	e2 = parameters[4]
	eps = 1e-1
	ps_list = []
	SO3s_list = []
	gripper_SE3s = []

	# box
	# if abs(e1 - 0.2) < eps and abs(e2 - 0.2) < eps: 
	if e2 < 0.6: 
		# default linspace
		grid = torch.linspace(0, 1, n_gripper_box)
		grid_2D_x, grid_2D_y = torch.meshgrid(grid, grid)
		grid = grid.reshape(-1, 1)
		grid_2D_x = grid_2D_x.reshape(-1, 1)
		grid_2D_y = grid_2D_y.reshape(-1, 1)
		# side grasp
		if a2 < max_width / 2:
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[a1 + d, 0, -a3 * ratio]])
			theta = torch.tensor(torch.pi)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[-a1 - d, 0, -a3 * ratio]])
			theta = torch.tensor(0)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
		if a1 < max_width / 2:
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[0, a2 + d, -a3 * ratio]])
			theta = torch.tensor(torch.pi / 2)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)
			ps = grid @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + torch.tensor([[0, -a2 - d, -a3 * ratio]])
			theta = torch.tensor(-torch.pi / 2)
			SO3s = torch.tensor([[
				[0, torch.sin(theta), torch.cos(theta)],
				[0, torch.cos(theta), -torch.sin(theta)],
				[-1, 0, 0]
			]]).repeat(n_gripper_box, 1, 1)
			ps_list.append(ps)
			SO3s_list.append(SO3s)   			
		
	# cylinder
	# elif abs(e1 - 0.2) < eps and abs(e2 - 1.0) < eps and abs(a2 - a1) < eps:
	else:
		# default linspace
		grid_r, grid_h = torch.linspace(0, 1, n_gripper_cyl_r), torch.linspace(0, 1, n_gripper_cyl_h)
		grid_2D_x, grid_2D_y = torch.meshgrid(grid_r, grid_h)
		grid_2D_x = grid_2D_x.reshape(-1, 1)
		grid_2D_y = grid_2D_y.reshape(-1, 1)
		# side grasp
		if a2 < max_width / 2:
			ps = torch.cos(2 * torch.pi * grid_2D_x) @ torch.tensor([[a1 + d, 0, 0]]) + \
				torch.sin(2 * torch.pi * grid_2D_x) @ torch.tensor([[0, a2 + d, 0]]) + \
				grid_2D_y @ torch.tensor([[0, 0, a3 * 2 * ratio]]) + \
				torch.tensor([[0, 0, -a3 * ratio]])
			SO3s = torch.zeros((n_gripper_cyl_h * n_gripper_cyl_r, 3, 3))		
			SO3s[:, 0, 1] = torch.sin(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 0, 2] = - torch.cos(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 1, 1] = - torch.cos(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 1, 2] = - torch.sin(2 * torch.pi * grid_2D_x).reshape(-1)
			SO3s[:, 2, 0] = -1
			ps_list.append(ps)
			SO3s_list.append(SO3s)   

	# # other shapes
	# else:
	# 	raise NotImplementedError

	# get gripper SE3s
	if len(ps_list) >= 1:
		gripper_ps = torch.cat(ps_list, dim=0)
		gripper_SO3s = torch.cat(SO3s_list, dim=0)
		gripper_SE3s = get_SE3s_torch(gripper_SO3s, gripper_ps)
		if len(SE3.shape) == 2:
			gripper_SE3s = SE3 @ gripper_SE3s.to(SE3)
		elif len(SE3.shape) == 3:
			gripper_SE3s = SE3.unsqueeze(1) @ gripper_SE3s.to(SE3).unsqueeze(0) # n_pose x n_grasp x 4 x 4
	else:
		return torch.tensor([])

	# grasp pose augment
	if flip:
		flip_matrix = torch.tensor([[
			[-1, 0, 0, 0], 
			[0, -1, 0, 0], 
			[0, 0, 1, 0], 
			[0, 0, 0, 1]
		]]).to(gripper_SE3s)
		if len(SE3.shape) == 3:
			flip_matrix = flip_matrix.unsqueeze(0)
		flipped_gripper_SE3s= gripper_SE3s @ flip_matrix
		if augment_flip:
			gripper_SE3s = torch.cat([gripper_SE3s, flipped_gripper_SE3s], dim=-3)
		else:
			gripper_SE3s = flipped_gripper_SE3s
		
	if tilt:
		theta = torch.tensor(torch.pi/6)
		tilt_matrix = torch.tensor([[
			[torch.cos(theta),  0, -torch.sin(theta),  d*torch.sin(theta)], 
			[0,	     			1,  0,  			   0], 
			[torch.sin(theta),  0,  torch.cos(theta),  d-d*torch.cos(theta)], 
			[0, 	 			0,  0,  			   1]
		]]).to(gripper_SE3s)
		if len(SE3.shape) == 3:
			tilt_matrix = tilt_matrix.unsqueeze(0)
		tilted_gripper_SE3s = gripper_SE3s @ tilt_matrix
		if augment_tilt:
			gripper_SE3s = torch.cat([gripper_SE3s, tilted_gripper_SE3s], dim=-3)
		else:
			gripper_SE3s = tilted_gripper_SE3s
   
	projected_z_axis_of_gripper = deepcopy(gripper_SE3s[...,0:3,2])
	projected_z_axis_of_gripper[..., 2] = 0
	projected_z_axis_of_gripper = projected_z_axis_of_gripper/projected_z_axis_of_gripper.norm(dim=-1, keepdim=True)
	bool = projected_z_axis_of_gripper @ desired_dir.to(gripper_SE3s) >= 0.5 ** 0.5 # 0.5
	#print(projected_z_axis_of_gripper @ desired_dir.to(gripper_SE3s))
	if return_bool:
		return gripper_SE3s[bool], bool
	else:
		return gripper_SE3s[bool]
