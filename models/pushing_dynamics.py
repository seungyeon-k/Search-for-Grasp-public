import torch
import torch.nn as nn

from loss.motion_prediction_loss import MotionPredictionLoss
from functions.lie_torch import (
	quats_to_matrices_torch,
	quats_to_matrices_torch_stable,
	matrices_to_quats_torch,
	matrices_to_quats_torch_stable,
)
from copy import deepcopy

from functions.LieGroup_torch import expSO3

class PushingDynamics(nn.Module):
	def __init__(
		self,
		module
	):
		super(PushingDynamics, self).__init__()
		self.module = module
		self.loss = MotionPredictionLoss(
			motion_dim=self.module.motion_dim,
			weight=self.module.ori_weight
		)

		# for debugging
		self.visualize_gdots = False

	def forward(self, x, a):
		y = self.module(x, a)
		return y

	def train_step(self, x_, y, optimizer, **kwargs):
		
		# get data
		x = x_[0]
		a = x_[1]

		# get processed input
		if self.visualize_gdots:
			g_dot_x, g_dot_a = self.get_input_after_group_action(x, a)

		# update
		optimizer.zero_grad()
		pred = self(x, a)
		loss = self.loss(y, pred)
		loss.backward()
		optimizer.step()

		# to numpy
		x_numpy = x.detach().cpu().numpy()
		a_numpy = a.detach().cpu().numpy()
		pred_numpy = pred.detach().cpu().numpy()
		y_numpy = y.detach().cpu().numpy()
		if self.visualize_gdots:
			g_dot_x_numpy = g_dot_x.detach().cpu().numpy()
			g_dot_a_numpy = g_dot_a.detach().cpu().numpy()

		results = {
			"loss": loss.item(),
			"prediction*": [x_numpy, a_numpy, pred_numpy],
			"ground_truth*": [x_numpy, a_numpy, y_numpy],
		}
		if self.visualize_gdots:
			results["g_dot_prediction*"] = [g_dot_x_numpy, g_dot_a_numpy, pred_numpy]

		return results

	def validation_step(self, x_, y, **kwargs):

		# get data
		x = x_[0]
		a = x_[1]

		# get processed input
		if self.visualize_gdots:
			g_dot_x, g_dot_a = self.get_input_after_group_action(x, a)

		# get validation loss
		pred = self(x, a)
		loss = self.loss(y, pred)
		loss.backward()

		# to numpy
		x_numpy = x.detach().cpu().numpy()
		a_numpy = a.detach().cpu().numpy()
		pred_numpy = pred.detach().cpu().numpy()
		y_numpy = y.detach().cpu().numpy()
		if self.visualize_gdots:
			g_dot_x_numpy = g_dot_x.detach().cpu().numpy()
			g_dot_a_numpy = g_dot_a.detach().cpu().numpy()

		results = {
			"loss": loss.item(),
			"val_prediction*": [x_numpy, a_numpy, pred_numpy],
			"val_ground_truth*": [x_numpy, a_numpy, y_numpy],
		}
		if self.visualize_gdots:
			results["val_g_dot_prediction*"] = [g_dot_x_numpy, g_dot_a_numpy, pred_numpy]

		return results

class EquivariantPushingDynamics(PushingDynamics):
	def __init__(self, module, plug_in_type):
		super(EquivariantPushingDynamics, self).__init__(module)
		self.plug_in_type = plug_in_type

		if ('ohp4' in self.plug_in_type) and ('ohs1' in self.plug_in_type):
			raise ValueError('ohp4 and ohs1 cannot coexist. check "plug_in_type" list in config file.')

	def unflatten_input(self, x, a):
		
		# load data
		bs = x.shape[0]
		no = x.shape[-1]
		confidences = x[:, :1, :].permute(0,2,1)
		positions = x[:, 1:4, :].permute(0,2,1)
		orientations = x[:, 4:8, :].permute(0,2,1)
		parameters = x[:, 8:, :].permute(0,2,1)
		a_positions = a[:, :3]
		a_vectors = a[:, 3:]

		# get SE3
		T = x.new_zeros((bs, no, 4, 4))
		R = quats_to_matrices_torch_stable(
			orientations
		)
		T[:, :, :3, :3] = R
		T[:, :, :3, 3] = positions
		T[:, :, 3, 3] = 1.0

		return confidences, T, parameters, a_positions, a_vectors

	def flatten_input(self, confidences, T, parameters, a_positions, a_vectors):
		
		# load data
		bs = T.shape[0]
		no = T.shape[1]
		confidences = confidences.permute(0,2,1)
		positions = T[:, :, :3, 3].permute(0,2,1)
		orientations = matrices_to_quats_torch_stable(
			T[:, :, :3, :3]
		).permute(0,2,1)
		parameters = parameters.permute(0,2,1)

		# get flattened data
		x = torch.cat(
			(confidences, positions, orientations, parameters),
			dim=1
		)
		a = torch.cat(
			(a_positions, a_vectors),
			dim=1
		) 

		return x, a

	def unflatten_output(self, y):
			
		# load data
		bs = y.shape[0]

		if self.module.motion_dim == '2D':
			theta = y[:, 2]
			position = y[:, :2]

			# get SE3
			deltaT = y.new_zeros((bs, 4, 4))
			deltaT[:, 0, 0] = torch.cos(theta)
			deltaT[:, 0, 1] = -torch.sin(theta)
			deltaT[:, 1, 0] = torch.sin(theta)
			deltaT[:, 1, 1] = torch.cos(theta)
			deltaT[:, :2, 3] = position
			deltaT[:, 2, 2] = 1.0
			deltaT[:, 3, 3] = 1.0

		elif self.module.motion_dim == '3D':
			position = y[:, :3]
			orientations = y[:, 3:]
			R = quats_to_matrices_torch_stable(
				orientations
			)

			# get SE3
			deltaT = y.new_zeros((bs, 4, 4))
			deltaT[:, :3, :3] = R
			deltaT[:, :3, 3] = position
			deltaT[:, 3, 3] = 1.0			
			
		else:
			raise ValueError('check the motion dimension in module!')

		return deltaT

	def flatten_output(self, deltaT):
		
		# get y
		if self.module.motion_dim == '2D':
			position = deltaT[:, :2, 3]
			orientation = torch.atan2(deltaT[:, 1, 0], deltaT[:, 0, 0])
			y = torch.cat(
				(position, orientation.unsqueeze(-1)),
				dim=1
			)
		elif self.module.motion_dim == '3D':
			position = deltaT[:, :3, 3]
			orientation = matrices_to_quats_torch_stable(
				deltaT[:, :3, :3]
			)
			y = torch.cat(
				(position, orientation),
				dim=1
			)

		else:
			raise ValueError('check the motion dimension in module!')

		return y
		
	def equivariant_plug_in_se2(self, T, parameters, a_positions, a_vectors):

		# reshape
		bs = T.shape[0]
		no = T.shape[1]
		parameters = parameters.reshape(-1, 5)
		T_target = T[:, 0, :, :]

		# initialize
		C = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(T)

		# rotation matrix
		z_axis = T_target[:, :3, 2]
		z_global_axis = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(len(z_axis), 1).to(T)
		cross_product = torch.cross(
			z_axis, 
			z_global_axis
		)
		sin_theta = torch.norm(cross_product, dim=1)
		cos_theta = torch.sum(z_axis * z_global_axis, dim=1)
		
		# case classification (for numerical stability)
		indices_1 = (sin_theta < 1e-12) * (cos_theta > 0)
		indices_2 = (sin_theta < 1e-12) * (cos_theta < 0)
		indices_3 = (~indices_1) * (~indices_2)

		# get rotation matrix for each case
		C[indices_1, :3, :3] = T_target[indices_1, :3, :3]
		permute_tensor = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 1]]).unsqueeze(0).to(T)
		C[indices_2, :3, :3] = -T_target[indices_2, :3, :3] @ permute_tensor
		screw = cross_product[indices_3] / sin_theta[indices_3].unsqueeze(-1)
		theta = torch.atan2(
			sin_theta[indices_3], cos_theta[indices_3]
		).unsqueeze(-1).unsqueeze(-1)
		M = expSO3(theta * torch.inverse(T_target[indices_3, :3, :3]) @ screw.unsqueeze(-1))
		C[indices_3, :3, :3] = T_target[indices_3, :3, :3] @ M			

		# translation matrix
		C[:, :2, 3] = T_target[:, :2, 3]

		return C

	def equivariant_plug_in_sym(self, T, parameters, a_positions, a_vectors):
		
		# reshape
		bs = T.shape[0]
		no = T.shape[1]
		a_dim = a_vectors.shape[-1]
		parameters = parameters.reshape(-1, 5)
		T = T.reshape(-1, 4, 4)
		a_vectors = a_vectors.unsqueeze(1).repeat(1, no, 1)
		a_vectors = a_vectors.reshape(-1, a_dim)
				
		# initialize
		D = torch.zeros(len(T), 3, 3).to(T)
		range_indices = torch.tensor(list(range(len(T)))).to(T).long()

		# shape classification
		indices_box = torch.abs(parameters[:, 3] - parameters[:, 4]) < 1e-3
		indices_cylinder = ~indices_box

		# align z-axis (box)
		T_box = T[indices_box, :, :]
		z_global_axis = torch.tensor([0, 0, 1]).unsqueeze(0).unsqueeze(-1).repeat(len(T_box), 1, 3).to(T)
		inner_product_z_box = torch.sum(
			z_global_axis * T_box[:, :3, :3], dim=1
		)
		_, new_z_axis_idx = torch.max(
			torch.abs(inner_product_z_box), dim=1
		)
		z_axis_sign = torch.sign(
			inner_product_z_box[range(len(new_z_axis_idx)), new_z_axis_idx]
		)
		D[range_indices[indices_box], new_z_axis_idx, 2] = z_axis_sign

		# align x-axis (box)
		action_box = a_vectors[indices_box, :]
		action_axis = torch.cat(
			(action_box, torch.zeros(len(T_box), 1).to(T)),
			dim=1
		).unsqueeze(-1).repeat(1, 1, 3)
		inner_product_x_box = torch.sum(
			action_axis * T_box[:, :3, :3], dim=1
		)
		_, new_x_axis_idx = torch.max(
			torch.abs(inner_product_x_box), dim=1
		)			
		x_axis_sign = torch.sign(
			inner_product_x_box[range(len(new_x_axis_idx)), new_x_axis_idx]
		)
		D[range_indices[indices_box], new_x_axis_idx, 0] = x_axis_sign	

		# align z-axis (cylinder)
		T_cylinder = T[indices_cylinder, :, :]
		z_global_axis = torch.tensor([0, 0, 1]).unsqueeze(0).repeat(len(T_cylinder), 1).to(T)
		inner_product_zz_cylinder = torch.sum(
			z_global_axis * T_cylinder[:, :3, 2], dim=1
		)
		z_axis_sign = torch.sign(inner_product_zz_cylinder)
		new_z_axis_idx = torch.ones(len(z_axis_sign)).long() * 2
		D[range_indices[indices_cylinder], new_z_axis_idx, 2] = z_axis_sign

		# align x-axis (cylinder)
		if 'ohp4' in self.plug_in_type:
			action_cylinder = a_vectors[indices_cylinder, :]
			action_axis = torch.cat(
				(action_cylinder, torch.zeros(len(T_cylinder), 1).to(T)),
				dim=1
			).unsqueeze(-1).repeat(1, 1, 3)
			inner_product_x_cylinder = torch.sum(
				action_axis * T_cylinder[:, :3, :3], dim=1
			)
			_, new_x_axis_idx = torch.max(
				torch.abs(inner_product_x_cylinder), dim=1
			)			
			x_axis_sign = torch.sign(
				inner_product_x_cylinder[range(len(new_x_axis_idx)), new_x_axis_idx]
			)
			D[range_indices[indices_cylinder], new_x_axis_idx, 0] = x_axis_sign		


		elif 'ohs1' in self.plug_in_type:
			action_cylinder = a_vectors[indices_cylinder, :]
			action_axis = torch.cat(
				(action_cylinder, torch.zeros(len(T_cylinder), 1).to(T)),
				dim=1
			)
			inner_product_xx_cylinder = torch.sum(
				T_cylinder[:, :3, 0] * action_axis, dim=1
			)
			cross_product_xx_cylinder = torch.cross(
				T_cylinder[:, :3, 0], action_axis
			)	
			sign_cylinder = torch.sign(
				torch.sum(T_cylinder[:, :3, 2] * cross_product_xx_cylinder, dim=1)
			)
			x_theta = torch.atan2(
				torch.norm(cross_product_xx_cylinder, dim=1), 
				inner_product_xx_cylinder
			) * sign_cylinder
			D[range_indices[indices_cylinder], 0, 0] = torch.cos(x_theta)
			D[range_indices[indices_cylinder], 1, 0] = torch.sin(x_theta)	

		# obtain y axis of D matrix
		D[:, :, 1] = torch.cross(D[:, :, 2], D[:, :, 0])

		# reshape D
		D = D.reshape(bs, no, 3, 3)
		try:
			D = torch.inverse(D)
		except:
			D = torch.eye(3).repeat(bs, no, 1, 1).to(D.device)

		return D

	def group_action_on_input_se2(self, T, parameters, a_positions, a_vectors, C):
		
		# load SE2
		bs = T.shape[0]
		no = T.shape[1]
		C_inv = torch.inverse(C)
		C_inv_repeated = C_inv.unsqueeze(1).repeat(1, no, 1, 1)

		# transform
		T_new = C_inv_repeated @ T
		a_positions_new = (
			C_inv @ torch.cat(
				(a_positions, torch.ones(len(a_positions), 1).to(a_positions)),
				dim=1
			).unsqueeze(-1)
		).squeeze(-1)[:, :-1]
		a_vectors_new = (
			C_inv[:, :2, :2] @ a_vectors.unsqueeze(-1)
		).squeeze(-1)

		return T_new, parameters, a_positions_new, a_vectors_new

	def group_action_on_input_sym(self, T, parameters, a_positions, a_vectors, D):

		# load symmetry group
		D_inv = torch.inverse(D)

		# initialize
		T_new = deepcopy(T)
		parameters_new = deepcopy(parameters) 

		# transform
		T_new[:, :, :3, :3] = T[:, :, :3, :3] @ D_inv
		parameters_new[:, :, :3] = (
			parameters[:, :, :3].unsqueeze(-2) @ (D_inv ** 2)
		).squeeze(-2)

		return T_new, parameters_new, a_positions, a_vectors

	def group_action_on_output_sym(self, y, D):
		
		# load symmetry group
		D_target = D[:, 0, :, :]

		# transform
		y[:, :3, :3] = torch.inverse(D_target) @ y[:, :3, :3] @ D_target
		y[:, :3, 3] = (torch.inverse(D_target) @ y[:, :3, 3].unsqueeze(-1)).squeeze(-1)

		return y

	def group_action_on_output_se2(self, y, C):
	
		return y
	
	def forward(self, x, a):
		
		# unflatten input
		confidences, T, parameters, a_positions, a_vectors = self.unflatten_input(x, a)
		
		# equivariant plug-in and group action on input
		if ('ohp4' in self.plug_in_type) or ('ohs1' in self.plug_in_type):
			gbarx_sym = self.equivariant_plug_in_sym(T, parameters, a_positions, a_vectors)
			T, parameters, a_positions, a_vectors = self.group_action_on_input_sym(T, parameters, a_positions, a_vectors, gbarx_sym)
		if 'se2' in self.plug_in_type:
			gbarx_se2 = self.equivariant_plug_in_se2(T, parameters, a_positions, a_vectors)
			T, parameters, a_positions, a_vectors = self.group_action_on_input_se2(T, parameters, a_positions, a_vectors, gbarx_se2)

		# flatten input
		g_dot_x, g_dot_a = self.flatten_input(confidences, T, parameters, a_positions, a_vectors)

		# model output
		y = self.module(g_dot_x, g_dot_a)

		# unflatten output
		y = self.unflatten_output(y)

		# group action on output
		if ('ohp4' in self.plug_in_type) or ('ohs1' in self.plug_in_type):
			y = self.group_action_on_output_sym(y, gbarx_sym)
		if 'se2' in self.plug_in_type:
			y = self.group_action_on_output_se2(y, gbarx_se2)
		
		# flatten output
		y = self.flatten_output(y)

		return y

	#################################################
	############## debugging functions ##############
	#################################################

	# to visualize inputs after group action
	def get_input_after_group_action(self, x, a):
			
		# unflatten input
		confidences, T, parameters, a_positions, a_vectors = self.unflatten_input(x, a)
		
		# equivariant plug-in
		if ('ohp4' in self.plug_in_type) or ('ohs1' in self.plug_in_type):
			gbarx_sym = self.equivariant_plug_in_sym(T, parameters, a_positions, a_vectors)
		if 'se2' in self.plug_in_type:
			gbarx_se2 = self.equivariant_plug_in_se2(T, parameters, a_positions, a_vectors)

		# group action on input
		if ('ohp4' in self.plug_in_type) or ('ohs1' in self.plug_in_type):
			T, parameters, a_positions, a_vectors = self.group_action_on_input_sym(T, parameters, a_positions, a_vectors, gbarx_sym)
		if 'se2' in self.plug_in_type:
			T, parameters, a_positions, a_vectors = self.group_action_on_input_se2(T, parameters, a_positions, a_vectors, gbarx_se2)

		# flatten input
		g_dot_x, g_dot_a = self.flatten_input(confidences, T, parameters, a_positions, a_vectors)

		return g_dot_x, g_dot_a