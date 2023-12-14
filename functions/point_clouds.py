from turtle import down
import numpy as np
from sklearn.preprocessing import normalize
import open3d as o3d
import random

def noise_augmentation(pc, noise_std=0.001):
    '''
    input: point cloud (3, n)
    output: noisy point cloud (3, n)
    '''
    pc_wo_label = pc[:3, :]
    noise = np.random.uniform(-1, 1, size=pc_wo_label.shape)
    noise = normalize(noise, axis=0, norm='l2')

    scale = np.random.normal(loc=0, scale=noise_std, size=(1, pc_wo_label.shape[1]))
    scale = scale.repeat(pc_wo_label.shape[0], axis=0)

    pc[:3, :] = pc[:3, :] + noise * scale

    return pc

def normalize_pointcloud(
        pc, 
        label=None, 
        mode='total', 
        scale_normalize=True, 
        mean_xyz=None, 
        diagonal_len=None,
        multidim_diagonal=True):

    # target object
    if mode == 'total':
        pc_target = pc[:3, :]
    elif mode == 'target':
        pc_target = pc[:3, pc[3] == 1]
        if pc_target.shape[1] < 30:
            pc_target = pc[:3, :]

    # calculate mean xyz and diagonal length
    if mean_xyz is None:
        mean_xyz = np.mean(pc_target, axis=1)
    if diagonal_len is None:
        max_xyz = np.max(pc_target, axis=1)
        min_xyz = np.min(pc_target, axis=1)
        if scale_normalize:
            diagonal_len = np.linalg.norm(max_xyz-min_xyz)
        else:
            diagonal_len = 1
        
    # normalize point cloud
    pc[:3, :] -= np.expand_dims(mean_xyz, 1)
    if multidim_diagonal:
        pc[:3, :] /= np.expand_dims(diagonal_len, 1)
    else:
        pc[:3, :] /= diagonal_len
        
    # output list
    outputs = [pc, mean_xyz, diagonal_len]

    if label is not None:
        # position
        label[:3] -= mean_xyz
        label[:3] /= diagonal_len

        # size parameters
        label[6:9] /= diagonal_len

        # append label
        outputs.insert(1, label)

    return outputs

def downsample_pointcloud(num_downsample_pc, pc, rgb_pc=None, labels=None):
    
    # # voxel uniform sampling
    # pcd = o3d.geometry.Pointcloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # downsampled_pcd = pcd.voxel_down_sample(v_size)

    # output lists
    outputs = []

    # point cloud downsample
    down_idxs = random.sample(range(pc.shape[0]), num_downsample_pc)
    pc_down = pc[down_idxs]
    outputs.append(pc_down)

    # colored point cloud downsample
    if rgb_pc is not None:
        rgb_pc_down = rgb_pc[down_idxs]
        outputs.append(rgb_pc_down)

    # segmentation label downsample
    if labels is not None:
        labels_down = labels[down_idxs]
        outputs.append(labels_down)

    return outputs

def upsample_pointcloud(num_upsample_pc, pc, rgb_pc=None, labels=None):
    while pc.shape[0] < num_upsample_pc:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        partial_pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        idx = random.choice(range(pc.shape[0]))
        idx_nearest_points = list(partial_pcd_tree.search_knn_vector_3d(pcd.points[idx], 2)[1])
        pc_append = np.mean([pc[idx_nearest_points]], axis=1)
        pc = np.append(pc, pc_append, axis=0)
        
        if rgb_pc is not None:
            rgb_pc_append = np.mean([rgb_pc[idx_nearest_points]], axis=1)
            rgb_pc = np.append(rgb_pc, rgb_pc_append, axis=0)
        
        if labels is not None:
            idx_nearest_points.remove(idx)
            label_append = labels[idx_nearest_points]
            labels = np.append(labels, label_append)

    if rgb_pc is not None:
        if labels is not None:
            return pc, rgb_pc, labels
        else:
            return pc, rgb_pc
    else:
        if labels is not None:
            return pc, labels
        else:
            return pc
