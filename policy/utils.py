import torch
from copy import deepcopy
from functions.utils_wandb import sq_pc_from_parameters
from functions.lie import quats_to_matrices, get_SE3s
from functions.lie_torch import exp_so3

def get_pc_afterimage(pc, distance=0.2, directions=[[0, 0, 1]], number_of_points=1024):
    """get pointcloud of afterimage made by moving given pointcloud along given direction by given distance 

    Args:
        pc (n_pc x 3 tensor): _description_
        distance (float): _description_
        directions (n_dir x 3 tensor): _description_
        
    Returns:
        afterimages (n_dir x n_pc x 3 tensor)
    """
    num_image = 10
    afterimages = []

    for i in range(num_image):
        afterimages.append(deepcopy(pc).unsqueeze(0) + (directions.unsqueeze(1) * distance * i / (num_image-1)))

    afterimages = torch.cat(afterimages, dim=1)

    random_idx = torch.randperm(afterimages.shape[1])[:number_of_points]
    return afterimages[:,random_idx,:]

def get_pc_of_objects(objects_poses, objects_sq_params):
    """get pointclouds of given objects

    Args:
        objects_poses (n x 4 x 4 tensor): SE(3) pose matrices of given objects
        objects_sq_params (n x 5 tensor): superquadric parameters of given objects

    Returns:
        _type_: (n x n_pc x 3 tensor) : (n_pc x 3) pointclouds of (n) objects.
    """
    pc_of_objects = torch.stack(
    [
        torch.from_numpy(
            sq_pc_from_parameters(objects_pose, objects_sq_param)
        ).float()
        for objects_pose, objects_sq_param
        in
        zip(objects_poses, objects_sq_params)
    ]
    ).to(objects_poses.device).float()
    return pc_of_objects

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

def get_shelf_sq_values(shelf, exclude_list=[], device=None):
    		
	# list
	mesh_list = []

	# get shelf parts
	part_keys = [key for key in shelf.keys() if key.startswith('shelf')]

	# get position and orientation
	global_position = shelf['global_position']
	global_orientation = shelf['global_orientation']
	global_SO3 = quats_to_matrices(global_orientation)

	# initialize
	Ts_shelf = []
	parameters_shelf = []

	for part_key in part_keys:
			
		# exclude
		if part_key.split('_')[-1] in exclude_list:
			continue

		# load info	
		size = shelf[part_key]['size']
		position = shelf[part_key]['position']
		
		# transform
		part_SO3 = global_SO3
		part_position = global_position + global_SO3.dot(position)
		T = get_SE3s(part_SO3, part_position)
		T = torch.tensor(T).float()

		# parameters
		parameter = torch.tensor([size[0]/2, size[1]/2, size[2]/2, 0.2, 0.2])

		# append
		Ts_shelf.append(T)
		parameters_shelf.append(parameter)	

	Ts_shelf = torch.stack(Ts_shelf)
	parameters_shelf = torch.stack(parameters_shelf)

	Ts_shelf = Ts_shelf.to(device)
	parameters_shelf = parameters_shelf.to(device)

	return Ts_shelf, parameters_shelf


