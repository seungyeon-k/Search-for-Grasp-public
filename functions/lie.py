import numpy as np
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


# Matrix to vector
def ToVector(mat):
    if np.size(mat, 1) == 4:          
        vec = np.zeros((6,1))
        vec[0] = -mat[1,2]
        vec[1] = mat[0,2]
        vec[2] = -mat[0,1]
        vec[3] = mat[0,3]
        vec[4] = mat[1,3]
        vec[5] = mat[2,3]
    elif np.size(mat, 1) == 3:     
        vec = np.zeros((3,1))
        vec[0] = -mat(1,2)
        vec[1] = mat(0,2)
        vec[2] = -mat(0,1)
    else:
        raise ValueError('Dimension is not 3 by 3 or 4 by 4')
        
    return vec


# skew matrix
def skew(w):
    W = np.array([[0, -w[2], w[1]],
                  [w[2], 0, -w[0]],
                  [-w[1], w[0], 0]])
    
    return W


# SO3 exponential
def exp_so3(w):
    if len(w) != 3:
        raise ValueError('Dimension is not 3')
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        R = np.eye(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        R = np.eye(3) + sw * wnorm_inv * W + (1 - cw) * np.power(wnorm_inv,2) * W.dot(W)

    return R


# SE3 exponential
def exp_se3(S):
    if len(S) != 6:
        raise ValueError('Dimension is not 6')
    w = S[0:3]
    v = S[3:6]
    eps = 1e-14
    wnorm = np.sqrt(sum(w*w))
    if wnorm < eps:
        T = np.eye(4)
        T[0:3,3] = v.reshape(3)
    else:
        wnorm_inv = 1 / wnorm
        cw = np.cos(wnorm)
        sw = np.sin(wnorm)
        W = skew(w)
        P = np.eye(3) + (1 - cw) * np.power(wnorm_inv,2) * W + (wnorm - sw) * np.power(wnorm_inv,3) * W.dot(W)
        T = np.eye(4)
        T[0:3,0:3] = exp_so3(w)
        T[0:3,3] = P.dot(v).reshape(3)
    
    return T


# Logarithm of SO3
def log_SO3(R):
    w = np.zeros((3,1))
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(cos_theta)
    if np.abs(theta) < 1e-6 :
        w = np.zeros((3,1))
    else :
        if abs(theta - np.pi) < 1e-6 : 
            for k in range(3) :
                if abs(1 + R[k,k]) > 1e-6 :
                    break
            w = deepcopy(R[:,k])
            w[k] = w[k] + 1
            w = w / np.sqrt(2 * (1 + R[k,k])) * theta
        else : 
            w_hat = (R - R.transpose()) / (2 * np.sin(theta)) * theta
            w[0] = w_hat[2,1]
            w[1] = w_hat[0,2]
            w[2] = w_hat[1,0]
            
    return w


# Logarithm of SE3
def log_SE3(T):

    R = T[0:3,0:3]
    p = T[0:3,3]
    logT = np.zeros((4,4))
    # if np.trace(R) < -1:
        # print(np.trace(R))
    cos_theta = (np.trace(R) - 1) / 2
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-6 :
        logT[0:3,3] = p
    else:
        w = log_SO3(R)
        W = skew(w)
        Pinv = np.eye(3) - 1 / 2 * theta * W + (1 - theta / 2 / np.tan(theta / 2)) * W.dot(W)
        logT[0:3,0:3] = W
        logT[0:3,3] = Pinv.dot(p)
  
    return logT


def quats_to_matrices(quats):    
    if quats.ndim == 1:
        matrices = R.from_quat(quats).as_matrix()
    elif quats.ndim == 2:
        try:
            matrices = np.array([R.from_quat(quats[i]).as_matrix() for i in range(len(quats))])
        except:
            print(quats)
    else:
        raise NotImplementedError("Dimension of quaternions must be 1 or 2")

    return matrices


def matrices_to_quats(matrices):
    if matrices.ndim == 2:
        quats = R.from_matrix(matrices).as_quat()
    elif matrices.ndim == 3:
        quats = np.array([R.from_matrix(matrices[i]).as_quat() for i in range(len(matrices))])
    else:
        raise NotImplementedError("Dimension of matrices must be 2 or 3")

    return quats


def get_SE3s(Rs, ps):
    assert Rs.ndim == ps.ndim + 1, f"Dimension of positions must be {Rs.ndim-1} if dimension of matrices is {Rs.ndim}"

    if Rs.ndim == 2:
        SE3s = np.identity(4)
        SE3s[:3, :3] = Rs
        SE3s[:3, 3] = ps
    elif Rs.ndim == 3:
        SE3s = np.repeat(np.expand_dims(np.identity(4), axis=0), len(Rs), axis=0)
        SE3s[:, :3, :3] = Rs
        SE3s[:, :3, 3] = ps
    else:
        raise NotImplementedError("Dimension of matrices must be 2 or 3")
    
    return SE3s