import numpy as np
import torch

def sq_distance(x, poses, sq_params, mode='e1'):
	'''
	input: x : (n x 3) pointcloud coordinates 
		   poses : (n_sq x 4 x 4) superquadric poses
		   sq_params : (n_sq x 5) superquadric parameters
		   mode : 'e1' or '1'
	output: pointcloud sdf values for each superquadrics (n x n_sq)
	'''    

	# parameters
	n_sq = len(sq_params)
	a1 = sq_params[:, [0]] 
	a2 = sq_params[:, [1]] 
	a3 = sq_params[:, [2]] 
	e1 = sq_params[:, [3]]
	e2 = sq_params[:, [4]]

	# object positions
	positions = poses[:, 0:3, 3] 
	rotations = poses[:, 0:3, 0:3] 

	# repeat voxel coordinates
	x = x.unsqueeze(0).repeat(n_sq, 1, 1).transpose(1,2) # (n_sq x 3 x n)

	# coordinate transformation
	rotations_t = rotations.permute(0,2,1)
	x_transformed = (
		- rotations_t @ positions.unsqueeze(2) 
		+ rotations_t @ x
	) # (n_sq x 3 x n) 

	# coordinates
	X = x_transformed[:, 0, :]
	Y = x_transformed[:, 1, :]
	Z = x_transformed[:, 2, :]

	# calculate beta
	F = (
		torch.abs(X/a1)**(2/e2)
		+ torch.abs(Y/a2)**(2/e2)
		)**(e2/e1) + torch.abs(Z/a3)**(2/e1)

	if mode == 'e1':
		F = F ** e1

	return F.T


def get_SQ_parameters_mujoco(shapes, sizes):
    n_shapes = len(shapes)
    parameters = np.zeros((n_shapes, 5))
    for idx, shape in enumerate(shapes):
        if shape == 'box':
            parameters[idx, 0] = sizes[idx, 0]   # a1
            parameters[idx, 1] = sizes[idx, 1]   # a2
            parameters[idx, 2] = sizes[idx, 2]   # a3
            parameters[idx, 3] = 0.2           # e1
            parameters[idx, 4] = 0.2           # e2
        elif shape == 'cylinder':
            parameters[idx, 0] = sizes[idx, 0]       # a1
            parameters[idx, 1] = sizes[idx, 0]       # a2
            parameters[idx, 2] = sizes[idx, 1]   # a3
            parameters[idx, 3] = 0.2           # e1
            parameters[idx, 4] = 1             # e2

    return parameters[:, 3:5], parameters[:, :3]

def get_SQ_parameters(object_infos):
    n_shapes = len(object_infos)
    parameters = np.zeros((n_shapes, 5))

    for object_id, object_info in enumerate(object_infos):
        if object_info['type'] == 'box':
            parameters[object_id, 0] = object_info['size'][0] / 2   # a1
            parameters[object_id, 1] = object_info['size'][1] / 2   # a2
            parameters[object_id, 2] = object_info['size'][2] / 2   # a3
            parameters[object_id, 3] = 0.2                          # e1
            parameters[object_id, 4] = 0.2                          # e2
        elif object_info['type'] == 'cylinder':
            parameters[object_id, 0] = object_info['size'][0]       # a1
            parameters[object_id, 1] = object_info['size'][1]       # a2
            parameters[object_id, 2] = object_info['size'][2] / 2   # a3
            parameters[object_id, 3] = 0.2                          # e1
            parameters[object_id, 4] = 1                            # e2
        else:
            raise NotImplementedError

    return parameters[:, 3:5], parameters[:, :3]