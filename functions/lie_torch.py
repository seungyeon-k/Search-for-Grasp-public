import torch
import torch.nn.functional as F

def quats_to_matrices_torch(quaternions):
	# An unit quaternion is q = xi + yj + zk + w
	x = quaternions[:, 0]
	y = quaternions[:, 1]
	z = quaternions[:, 2]
	w = quaternions[:, 3]

	# Initialize
	K = quaternions.shape[0]
	R = quaternions.new_zeros((K, 3, 3))

	xx = x**2
	yy = y**2
	zz = z**2
	ww = w**2
	n = (ww + xx + yy + zz).unsqueeze(-1)
	s = quaternions.new_zeros((K, 1))
	s[n != 0] = 2 / n[n != 0]

	xy = s[:, 0] * x * y
	xz = s[:, 0] * x * z
	xw = s[:, 0] * x * w
	yz = s[:, 0] * y * z
	yw = s[:, 0] * y * w
	zw = s[:, 0] * z * w

	xx = s[:, 0] * xx
	yy = s[:, 0] * yy
	zz = s[:, 0] * zz

	idxs = torch.arange(K).to(quaternions.device)
	R[idxs, 0, 0] = 1 - yy - zz
	R[idxs, 0, 1] = xy - zw
	R[idxs, 0, 2] = xz + yw

	R[idxs, 1, 0] = xy + zw
	R[idxs, 1, 1] = 1 - xx - zz
	R[idxs, 1, 2] = yz - xw

	R[idxs, 2, 0] = xz - yw
	R[idxs, 2, 1] = yz + xw
	R[idxs, 2, 2] = 1 - xx - yy

	return R

def quats_to_matrices_torch_stable(quaternions):
	"""
	Convert rotations given as quaternions to rotation matrices.
	Args:
		quaternions: quaternions with real part last (original version is "first"),
			as tensor of shape (..., 4).
	Returns:
		Rotation matrices as tensor of shape (..., 3, 3).
	"""
	i, j, k, r = torch.unbind(quaternions, -1)
	# pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
	two_s = 2.0 / (quaternions * quaternions).sum(-1)

	o = torch.stack(
		(
			1 - two_s * (j * j + k * k),
			two_s * (i * j - k * r),
			two_s * (i * k + j * r),
			two_s * (i * j + k * r),
			1 - two_s * (i * i + k * k),
			two_s * (j * k - i * r),
			two_s * (i * k - j * r),
			two_s * (j * k + i * r),
			1 - two_s * (i * i + j * j),
		),
		-1,
	)
	return o.reshape(quaternions.shape[:-1] + (3, 3))

def thetas_to_matrices_torch(theta):
	# initialize
	K = theta.shape[0]
	R = theta.new_zeros((K, 3, 3))

	cos = theta[:,0]
	sin = theta[:,1]

	idxs = torch.arange(K).to(theta.device)
	R[idxs, 0, 0] = cos
	R[idxs, 0, 1] = -sin
	R[idxs, 1, 0] = sin
	R[idxs, 1, 1] = cos
	R[idxs, 2, 2] = 1

	return R


def matrices_to_quats_torch(R):
	original_ndim = R.ndim
	if original_ndim == 2:
		R = R.unsqueeze(0).to(R)
	elif original_ndim == 3:
		pass
	else:
		raise NotImplementedError("Dimension of matrices must be 2 or 3")

	qr = 0.5 * torch.sqrt(1+torch.einsum('ijj->i', R)).unsqueeze(1)
	qi = 1/(4*qr) * (R[:, 2,1] - R[:, 1,2]).unsqueeze(1)
	qj = 1/(4*qr) * (R[:, 0,2] - R[:, 2,0]).unsqueeze(1)
	qk = 1/(4*qr) * (R[:, 1,0] - R[:, 0,1]).unsqueeze(1)

	if original_ndim == 2:
		R = R.squeeze(0)

	return torch.cat([qi, qj, qk, qr], dim=1).to(R)

def matrices_to_quats_torch_stable(matrix):
	"""
	Convert rotations given as rotation matrices to quaternions.
	Args:
		matrix: Rotation matrices as tensor of shape (..., 3, 3).
	Returns:
		quaternions with real part last (original version is "first"), as tensor of shape (..., 4).
	"""
	if matrix.size(-1) != 3 or matrix.size(-2) != 3:
		raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

	batch_dim = matrix.shape[:-2]
	m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
		matrix.reshape(batch_dim + (9,)), dim=-1
	)

	q_abs = _sqrt_positive_part(
		torch.stack(
			[
				1.0 + m00 + m11 + m22,
				1.0 + m00 - m11 - m22,
				1.0 - m00 + m11 - m22,
				1.0 - m00 - m11 + m22,
			],
			dim=-1,
		)
	)

	# we produce the desired quaternion multiplied by each of r, i, j, k
	quat_by_rijk = torch.stack(
		[
			# pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
			#  `int`.
			torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
			# pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
			#  `int`.
			torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
			# pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
			#  `int`.
			torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
			# pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
			#  `int`.
			torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
		],
		dim=-2,
	)

	# We floor here at 0.1 but the exact level is not important; if q_abs is small,
	# the candidate won't be picked.
	flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
	quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

	# if not for numerical problems, quat_candidates[i] should be same (up to a sign),
	# forall i; we pick the best-conditioned one (with the largest denominator)

	temp = quat_candidates[
		F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
	].reshape(batch_dim + (4,)) # this is the original output

	return temp[..., [1, 2, 3, 0]]

def _sqrt_positive_part(x):
	"""
	Returns torch.sqrt(torch.max(0, x))
	but with a zero subgradient where x is 0.
	"""
	ret = torch.zeros_like(x)
	positive_mask = x > 0
	ret[positive_mask] = torch.sqrt(x[positive_mask])
	return ret


def get_device_info(x):
	cuda_check = x.is_cuda
	if cuda_check:
		device = "cuda:{}".format(x.get_device())
	else:
		device = 'cpu'
	return device


def skew(w):
	n = w.shape[0]
	device = get_device_info(w)
	if w.shape == (n, 3, 3):
		W = torch.cat([-w[:, 1, 2].unsqueeze(-1),
					   w[:, 0, 2].unsqueeze(-1),
					   -w[:, 0, 1].unsqueeze(-1)], dim=1)
	else:
		if w.ndim == 3:
			w = w.squeeze(-1)
		zero1 = torch.zeros(n, 1, 1).to(device)
		# zero1 = torch.zeros(n, 1, 1)
		w = w.unsqueeze(-1).unsqueeze(-1)
		W = torch.cat([torch.cat([zero1, -w[:, 2], w[:, 1]], dim=2),
					   torch.cat([w[:, 2], zero1, -w[:, 0]], dim=2),
					   torch.cat([-w[:, 1], w[:, 0], zero1], dim=2)], dim=1)
	return W


def exp_so3(Input):
	device = get_device_info(Input)
	n = Input.shape[0]
	if Input.shape == (n, 3, 3):
		W = Input
		w = skew(Input)
	else:
		w = Input
		W = skew(w)

	wnorm_sq = torch.sum(w * w, dim=1)
	wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)

	wnorm = torch.sqrt(wnorm_sq)
	wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)

	cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)
	sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)
	w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
	w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
	w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
	eps = 1e-7

	R = torch.zeros(n, 3, 3).to(device)

	R[wnorm > eps] = torch.cat((torch.cat((cw - ((w0 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
										   - (w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
										   (w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
										  dim=2),
								torch.cat(((w2 * sw) / wnorm_unsqueezed - (w0 * w1 * (cw - 1)) / wnorm_sq_unsqueezed,
										   cw - ((w1 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed,
										   - (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed),
										  dim=2),
								torch.cat((-(w1 * sw) / wnorm_unsqueezed - (w0 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
										   (w0 * sw) / wnorm_unsqueezed - (w1 * w2 * (cw - 1)) / wnorm_sq_unsqueezed,
										   cw - ((w2 ** 2) * (cw - 1)) / wnorm_sq_unsqueezed),
										  dim=2)),
							   dim=1)[wnorm > eps]

	R[wnorm <= eps] = torch.eye(3).to(device) + W[wnorm < eps] + 1 / 2 * W[wnorm < eps] @ W[wnorm < eps]
	return R


def exp_se3(S):
	device = get_device_info(S)
	n = S.shape[0]
	if S.shape == (n, 4, 4):
		S1 = skew(S[:, :3, :3]).clone()
		S2 = S[:, 0:3, 3].clone()
		S = torch.cat([S1, S2], dim=1)
	# shape(S) = (n,6,1)
	w = S[:, :3]  # dim= n,3
	v = S[:, 3:].unsqueeze(-1)  # dim= n,3
	wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
	wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1,1)
	wnorm = torch.sqrt(wsqr)  # dim = (n)
	wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim = (n,1,1)
	wnorm_inv = 1 / wnorm_unsqueezed  # dim = (n)
	cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)
	sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1,1)

	eps = 1e-014
	W = skew(w)
	P = torch.eye(3, device=device) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
	# P = torch.eye(3) + (1 - cw) * (wnorm_inv ** 2) * W + (wnorm_unsqueezed - sw) * (wnorm_inv ** 3) * torch.matmul(W, W)  # n,3,3
	P[wnorm < eps] = torch.eye(3, device=device)
	# P[wnorm < eps] = torch.eye(3)
	T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4, device=device))], dim=1)
	# T = torch.cat([torch.cat([exp_so3(w), P @ v], dim=2), (torch.zeros(n, 1, 4))], dim=1)
	T[:, -1, -1] = 1
	return T


def get_SE3s_torch(Rs, ps):
	assert Rs.ndim == ps.ndim + 1, f"Dimension of positions must be {Rs.ndim-1} if dimension of matrices is {Rs.ndim}"

	if Rs.ndim == 2:
		SE3s = torch.eye(4).to(Rs)
		SE3s[:3, :3] = Rs
		SE3s[:3, 3] = ps
	elif Rs.ndim == 3:
		SE3s = torch.cat([torch.eye(4).unsqueeze(0)] * len(Rs)).to(Rs)
		SE3s[:, :3, :3] = Rs
		SE3s[:, :3, 3] = ps
	else:
		raise NotImplementedError("Dimension of matrices must be 2 or 3")
	
	return SE3s

def vectorize_scene_data_torch(
		T, 
		params, 
		a_pos, 
		a_vec, 
		num_primitives=5, 
		motion_dim='2D'
	):

	# number of objects
	num_objects = len(T)

	# compute positions and orientations
	positions = T[:, :3, 3]
	orientation = matrices_to_quats_torch_stable(
		T[:, :3, :3]
	)    

	# scene vector
	x = torch.zeros((13, num_primitives)).to(T)
	x[:, :num_objects] = torch.cat(
		[
			torch.ones((1, num_objects)).to(T), 
			positions.T, 
			orientation.T, 
			params.T
		]
	)
	x[4, num_objects:] = 1 # for numerical issue

	# action vector
	a = torch.cat([a_pos, a_vec])

	# # displacement vector
	# if motion_dim == '2D':
	# 	position_diff = T_diff[:2, 3]
	# 	orientation_diff = np.array(
	# 		[np.arctan2(T_diff[1, 0], T_diff[0, 0])]
	# 	)
	# elif motion_dim == '3D':
	# 	position_diff = T_diff[:3, 3]
	# 	orientation_diff = matrices_to_quats_torch_stable(
	# 		T_diff[:3, :3]
	# 	)
	# y = np.concatenate([position_diff, orientation_diff])

	return x, a #, y