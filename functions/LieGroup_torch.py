import torch

torch.pi = torch.acos(torch.zeros(1)).item() * 2
angle_threshold = 1e-6


def getNullspace(tensor2dim):
    if tensor2dim.is_complex():
        print('ERROR : getNullspace() fed by a complex number')
        exit(1)
    U, S, V = torch.Tensor.svd(tensor2dim, some=False, compute_uv=True)
    # threshold = torch.max(S) * torch.finfo(S.dtype).eps * max(U.shape[0], V.shape[1])
    # rank = torch.sum(S > threshold, dtype=int)
    rank = len(S)
    # return V[rank:, :].T.cpu().conj()
    return V[:, rank:]


def revoluteTwist(twist):
    nJoint, mustbe6 = twist.shape
    if mustbe6 != 6:
        print(f'[ERROR] revoluteTwist: twist.shape = {twist.shape}')
        exit(1)
    w = twist[:, :3]
    v = twist[:, 3:]
    w_normalized = w / w.norm(dim=1).view(nJoint, 1)
    proejctedTwists = torch.empty_like(twist)
    proejctedTwists[:, :3] = w_normalized
    wdotv = torch.sum(v * w_normalized, dim=1).view(nJoint, 1)
    proejctedTwists[:, 3:] = v - wdotv * w_normalized
    return proejctedTwists


def skew3dim(vec3dim):
    return skew_so3(vec3dim)


def skew_so3(so3):
    nBatch = len(so3)
    if so3.shape == (nBatch, 3, 3):
        return torch.cat([-so3[:, 1, 2].unsqueeze(-1),
                          so3[:, 0, 2].unsqueeze(-1),
                          -so3[:, 0, 1].unsqueeze(-1)], dim=1)
    elif so3.numel() == nBatch * 3:
        w = so3.reshape(nBatch, 3, 1, 1)
        zeroBatch = so3.new_zeros(nBatch, 1, 1)
        output = torch.cat([torch.cat([zeroBatch, -w[:, 2], w[:, 1]], dim=2),
                            torch.cat([w[:, 2], zeroBatch, -w[:, 0]], dim=2),
                            torch.cat([-w[:, 1], w[:, 0], zeroBatch], dim=2)], dim=1)
        return output
    else:
        print(f'ERROR : skew_so3, so3.shape = {so3.shape}')
        exit(1)


def skew6dim(vec6dim):
    return skew_se3(vec6dim)


def skew_se3(se3):
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        output = se3.new_zeros(nBatch, 6)
        output[:, :3] = skew_so3(se3[:, :3, :3])
        output[:, 3:] = se3[:, :3, 3]
        return output
    elif se3.numel() == nBatch * 6:
        se3_ = se3.reshape(nBatch, 6)
        output = se3_.new_zeros(nBatch, 4, 4)
        output[:, :3, :3] = skew_so3(se3_[:, :3])
        output[:, :3, 3] = se3_[:, 3:]
        return output
    else:
        print(f'ERROR : skew_se3, se3.shape = {se3.shape}')
        exit(1)


def expSO3(so3):
    nBatch = len(so3)
    if so3.shape == (nBatch, 3, 3):
        so3mat = so3
        so3vec = skew_so3(so3)
    elif so3.numel() == nBatch * 3:
        so3mat = skew_so3(so3)
        so3vec = so3.reshape(nBatch, 3)
    else:
        print(f'ERROR : expSO3, so3.shape = {so3.shape}')
        exit(1)
    # Rodrigues' rotation formula
    theta = so3vec.norm(dim=1)
    wmat = skew_so3(so3vec / theta.unsqueeze(-1))
    expso3 = so3.new_zeros(nBatch, 3, 3)
    zeroID = abs(theta) == 0 # < eps
    if zeroID.any():
        expso3[zeroID, 0, 0] = expso3[zeroID, 1, 1] = expso3[zeroID, 2, 2] = 1
    if (~zeroID).any():
        nNonZero = (~zeroID).sum()
        _theta = theta[~zeroID].reshape(nNonZero, 1)
        _wmat = wmat[~zeroID]
        expso3[~zeroID, 0, 0] = expso3[~zeroID, 1, 1] = expso3[~zeroID, 2, 2] = 1
        expso3[~zeroID] += _theta.sin().view(nNonZero, 1, 1) * _wmat
        expso3[~zeroID] += (1 - _theta.cos()).view(nNonZero, 1, 1) * _wmat @ _wmat
    # expso3[:, 0, 0] = expso3[:, 1, 1] = expso3[:, 2, 2] = 1
    # expso3 += theta.sin().view(nBatch, 1, 1) * wmat
    # expso3 += (1 - theta.cos()).view(nBatch, 1, 1) * wmat @ wmat
    return expso3


def expSE3(se3, dim12=False):
    eps = 1e-7
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        se3mat = se3
        se3vec = skew_se3(se3)
    elif se3.numel() == nBatch * 6:
        se3mat = skew_se3(se3)
        se3vec = se3.reshape(nBatch, 6)
    else:
        print(f'ERROR : expSE3, se3.shape = {se3.shape}')
        exit(1)
    # normalize
    w = se3vec[:, :3]
    v = se3vec[:, 3:]
    theta = w.norm(dim=1)
    zeroID = abs(theta) == 0 # < eps
    expse3 = se3.new_zeros(nBatch, 4, 4)
    if zeroID.any():
        expse3[zeroID, 0, 0] = expse3[zeroID, 1, 1] = expse3[zeroID, 2, 2] = expse3[zeroID, 3, 3] = 1
        expse3[zeroID, :3, 3] = v[zeroID]
    if (~zeroID).any():
        nNonZero = (~zeroID).sum()
        _theta = theta[~zeroID].reshape(nNonZero, 1)
        wmat = skew_so3(w[~zeroID] / _theta)
        # G = eye * theta + (1-cos(theta)) [w] + (theta - sin(theta)) * [w]^2
        G = se3.new_zeros(nNonZero, 3, 3)
        G[:, 0, 0] = G[:, 1, 1] = G[:, 2, 2] = _theta.view(nNonZero)
        G += (1 - _theta.cos()).view(nNonZero, 1, 1) * wmat
        G += (_theta - _theta.sin()).view(nNonZero, 1, 1) * wmat @ wmat
        # output
        expse3[~zeroID, :3, :3] = expSO3(w[~zeroID])
        expse3[~zeroID, :3, 3] = (G @ (v[~zeroID] / _theta).view(nNonZero, 3, 1)).view(nNonZero, 3)
        expse3[~zeroID, 3, 3] = 1
    if dim12:
        expse3 = (expse3[:, :3]).reshape(nBatch, 12)
    return expse3
    # return torch.matrix_exp(se3mat)

def clipping(x, low=-1.0, high=1.0):
    eps = 1e-6
    x[x<=low] = low + eps
    x[x>=high] = high - eps
    return x

def logSO3(SO3):
    nBatch = len(SO3)
    trace = torch.einsum('xii->x', SO3)
    regularID = (trace + 1).abs() >= angle_threshold
    singularID = (trace + 1).abs() < angle_threshold
    theta = torch.acos(clipping((trace - 1) / 2)).view(nBatch, 1, 1)
    so3mat = SO3.new_zeros(nBatch, 3, 3)
    # regular
    if any(regularID):
        so3mat[regularID, :, :] = (SO3[regularID] - SO3[regularID].transpose(1, 2)) / (2 * theta[regularID].sin()) * theta[regularID]
    # singular
    if any(singularID):
        if all(SO3[singularID, 2, 2] != -1):
            r = SO3[singularID, 2, 2]
            w = SO3[singularID, :, 2]
            w[:, 2] += 1
        elif all(SO3[singularID, 1, 1] != -1):
            r = SO3[singularID, 1, 1]
            w = SO3[singularID, :, 1]
            w[:, 1] += 1
        elif all(SO3[singularID, 0, 0] != -1):
            r = SO3[singularID, 0, 0]
            w = SO3[singularID, :, 0]
            w[:, 0] += 1
        else:
            print(f'ERROR: all() is somewhat ad-hoc. should be fixed.')
            exit(1)
        so3mat[singularID, :, :] = skew_so3(torch.pi / (2 * (1 + r)).sqrt().view(-1, 1) * w)
    # trace == 3 (zero rotation)
    if any((trace - 3).abs() < 1e-10):
        so3mat[(trace - 3).abs() < 1e-10] = 0
    return so3mat


def logSE3(SE3):
    nBatch = len(SE3)
    trace = torch.einsum('xii->x', SE3[:, :3, :3])
    regularID = (trace - 3).abs() >= angle_threshold
    zeroID = (trace - 3).abs() < angle_threshold
    se3mat = SE3.new_zeros(nBatch, 4, 4)
    if any(zeroID):
        se3mat[zeroID, :3, 3] = SE3[zeroID, :3, 3]
    if any(regularID):
        nRegular = sum(regularID)
        so3 = logSO3(SE3[regularID, :3, :3])
        # theta = skew_so3(so3).norm(dim=1).reshape(nRegular,1,1)
        theta = (torch.acos(clipping(0.5*(trace[regularID]-1)))).reshape(nRegular, 1, 1)
        wmat = so3 / theta
        identity33 = torch.zeros_like(so3)
        identity33[:, 0, 0] = identity33[:, 1, 1] = identity33[:, 2, 2] = 1
        invG = (1 / theta) * identity33 - 0.5 * wmat + (1 / theta - 0.5 / (0.5 * theta).tan()) * wmat @ wmat
        se3mat[regularID, :3, :3] = so3
        se3mat[regularID, :3, 3] = theta.view(nRegular, 1) * (invG @ SE3[regularID, :3, 3].view(nRegular, 3, 1)).reshape(nRegular, 3)
    return se3mat


def largeAdjoint(SE3):
    # R     0
    # [p]R  R
    nBatch = len(SE3)
    if SE3.shape != (nBatch, 4, 4):
        print(f'ERROR : SE3.shape = {SE3.shape}')
        exit(1)
    R, p = SE3[:, :3, :3], SE3[:, :3, 3].unsqueeze(-1)
    Adj = SE3.new_zeros(nBatch, 6, 6)
    Adj[:, :3, :3] = Adj[:, 3:, 3:] = R
    Adj[:, 3:, :3] = skew_so3(p) @ R
    return Adj
    pass


def smallAdjoint(se3):
    # [w] 0
    # [v] [w]
    nBatch = len(se3)
    if se3.shape == (nBatch, 4, 4):
        se3vec = skew_se3(se3)
    elif se3.numel() == nBatch * 6:
        se3vec = se3.reshape(nBatch, 6)
    else:
        print(f'ERROR : smallAdjoint, se3.shape = {se3.shape}')
        exit(1)
    w = se3vec[:, :3]
    v = se3vec[:, 3:]
    smallad = se3.new_zeros(nBatch, 6, 6)
    smallad[:, :3, :3] = smallad[:, 3:, 3:] = skew_so3(w)
    smallad[:, 3:, :3] = skew_so3(v)
    return smallad


def invSE3(SE3):
    nBatch = len(SE3)
    R, p = SE3[:, :3, :3], SE3[:, :3, 3].unsqueeze(-1)
    invSE3_ = SE3.new_zeros(nBatch, 4, 4)
    invSE3_[:, :3, :3] = R.transpose(1, 2)
    invSE3_[:, :3, 3] = - (R.transpose(1, 2) @ p).view(nBatch, 3)
    invSE3_[:, 3, 3] = 1
    return invSE3_


def invSO3(SO3):
    nBatch = len(SO3)
    if SO3.shape != (nBatch, 3, 3):
        print(f'[ERROR] invSO3 : SO3.shape = {SO3.shape}')
        exit(1)
    return SO3.transpose(1, 2)


def quaternions_to_rotation_matrices_torch(quaternions):
    assert quaternions.shape[1] == 4

    # initialize
    K = quaternions.shape[0]
    R = quaternions.new_zeros((K, 3, 3))

    # A unit quaternion is q = w + xi + yj + zk
    xx = quaternions[:, 1] ** 2
    yy = quaternions[:, 2] ** 2
    zz = quaternions[:, 3] ** 2
    ww = quaternions[:, 0] ** 2
    n = (ww + xx + yy + zz).unsqueeze(-1)
    s = quaternions.new_zeros((K, 1))
    s[n != 0] = 2 / n[n != 0]

    xy = s[:, 0] * quaternions[:, 1] * quaternions[:, 2]
    xz = s[:, 0] * quaternions[:, 1] * quaternions[:, 3]
    yz = s[:, 0] * quaternions[:, 2] * quaternions[:, 3]
    xw = s[:, 0] * quaternions[:, 1] * quaternions[:, 0]
    yw = s[:, 0] * quaternions[:, 2] * quaternions[:, 0]
    zw = s[:, 0] * quaternions[:, 3] * quaternions[:, 0]

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


def rotation_matrices_to_quaternions_torch(R):
    qr = 0.5 * torch.sqrt(1 + torch.einsum('ijj->i', R)).unsqueeze(1)
    qi = 1 / (4 * qr) * (R[:, 2, 1] - R[:, 1, 2]).unsqueeze(1)
    qj = 1 / (4 * qr) * (R[:, 0, 2] - R[:, 2, 0]).unsqueeze(1)
    qk = 1 / (4 * qr) * (R[:, 1, 0] - R[:, 0, 1]).unsqueeze(1)
    return torch.cat([qr, qi, qj, qk], dim=1)

def SE3_12dim_to_mat(vec):
    original_shape = vec.shape
    num_traj = original_shape[0]
    if len(original_shape) == 2:
        num_points = int(original_shape[-1]/12)
    else:
        num_points = original_shape[1]
    vec = vec.reshape(-1, 3, 4)
    T = torch.zeros(len(vec), 4, 4).to(vec)
    T[:, :3] = vec
    T[:, -1, -1] = 1
    T = T.reshape(num_traj, num_points, 4, 4)
    return T
