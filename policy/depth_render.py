import numpy as np
import torch
import torch.nn as nn
import h5py
import matplotlib.pyplot as plt

from functions.superquadrics import sq_distance
from copy import deepcopy

import time

class DepthRenderer(nn.Module):
    def __init__(self, 
                 camera_param,
                 device,
                 load_camera_param=False,
                 near=0.,
                 far=1.5,
                 N_samples=10,
                 sharpness=1000,
                 tau=100,
                 reduce_ratio=7,
                 **kargs):
        super(DepthRenderer, self).__init__()

        # load camera parameters
        if load_camera_param:
            camera_param = h5py.File(camera_param, 'r')
            self.camera_image_size = camera_param['camera_image_size'][()]
            # intrinsic parameter
            self.K = torch.tensor(camera_param['camera_intr'][()])
            # extrinsic parameter
            camera_pose = camera_param['camera_pose'][()]
        else:
            self.camera_image_size = deepcopy(camera_param['camera_image_size'])
            # intrinsic parameter
            self.K = deepcopy(torch.tensor(camera_param['camera_intr']))
            # extrinsic parameter
            camera_pose = deepcopy(camera_param['camera_pose'])

        # image size
        self.height = self.camera_image_size[0]
        self.width = self.camera_image_size[1]
        
        camera_pose[:3, :3] = camera_pose[:3, :3].dot(
            np.array([[1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]])
        )
        self.camera_pose = torch.tensor(camera_pose)

        # rendering parameters
        self.near = near
        self.far = far
        self.N_samples = N_samples
        self.sharpness = sharpness
        self.tau = tau
        self.device = device
        self.reduce_ratio = reduce_ratio
        
        self.get_ray_points()
    
    def get_ray_points(self):
        with torch.no_grad():
            i, j = torch.meshgrid(
                torch.linspace(0, self.width-1, int(self.width//self.reduce_ratio)), torch.linspace(0, self.height-1, int(self.height//self.reduce_ratio))
            )  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            
            # sample ray directions
            dirs = torch.stack(
                [(i-self.K[0][2])/self.K[0][0], -(j-self.K[1][2])/self.K[1][1], -torch.ones_like(i)], 
                -1
            )
            dirs = dirs / torch.norm(dirs, dim=2).unsqueeze(2)

            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs[..., np.newaxis, :] * self.camera_pose[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

            # Translate camera frame's origin to the world frame. 
            # It is the origin of all rays.
            rays_o = self.camera_pose[:3,-1]
            
            self.rays_d = rays_d.to(self.device).float()
            self.rays_o = rays_o.to(self.device).float()
            self.t = torch.linspace(-1, 1, self.N_samples).to(self.device) # we will use self.t at get_valid_points

    def get_valid_points(self, sq_poses, sq_params):
        with torch.no_grad():
            sizes = sq_params[:, 0:3] # n x 3
            sq_orientations = sq_poses[:, 0:3, 0:3] # n x 3 x 3
            sq_positions = sq_poses[:, 0:3, 3] # n x 3
            # view rays at sq pose and resize coordinates so that sq_sizes are all 1.
            transformed_directions = (sq_orientations.permute(0, 2, 1).unsqueeze(1).unsqueeze(1) @ self.rays_d.unsqueeze(-1)).squeeze() # n x H x W x 3
            transformed_directions = transformed_directions / sizes.unsqueeze(1).unsqueeze(1) # n x H x W x 3
            transformed_directions = transformed_directions / transformed_directions.norm(dim=-1, keepdim=True)

            transformed_camera_pos = (sq_orientations.permute(0, 2, 1) @ (self.rays_o - sq_positions).unsqueeze(-1)).squeeze() # n x 3
            transformed_camera_pos = transformed_camera_pos / sizes # n x 3
            
            rays_H = torch.abs((transformed_camera_pos.unsqueeze(1).unsqueeze(1) * transformed_directions).sum(dim=-1, keepdim=True)) * transformed_directions # n x H x W x 3
            
            dist = (transformed_camera_pos.unsqueeze(1).unsqueeze(1) + rays_H).norm(dim=-1)

            mask = dist < 3 ** 0.5 # n x H x W, True when the ray points intersects the sphere surrounding superquadric
            half_chord_length = (3 - dist[mask] ** 2) ** 0.5 # len(mask)
            
            points = (transformed_camera_pos.unsqueeze(1).unsqueeze(1) + rays_H)[mask].unsqueeze(1) + (half_chord_length.unsqueeze(-1) * self.t.unsqueeze(0)).unsqueeze(-1) * transformed_directions[mask].unsqueeze(1) # len(mask) x N_samples x 3
            
            points = points * sizes.repeat_interleave(mask.sum(dim=[1,2]), dim=0).unsqueeze(1)
            points = (sq_orientations.repeat_interleave(mask.sum(dim=[1,2]), dim=0).unsqueeze(1) @ points.unsqueeze(-1)).squeeze() + sq_positions.repeat_interleave(mask.sum(dim=[1,2]), dim=0).unsqueeze(1)

            points = torch.cat(
                [
                self.rays_o + (self.rays_d.repeat(len(sizes), 1, 1, 1)[mask] * self.near).unsqueeze(1),
                points,
                self.rays_o + (self.rays_d.repeat(len(sizes), 1, 1, 1)[mask] * self.far).unsqueeze(1)
                ],
                dim=1
                )
        return mask, points
        
    def get_all_depths(self, sq_poses, sq_params):
        """
        input: sq_poses (n x 4 x 4 torch tensor)
               sq_params (n x 5 torch tensor)
        output: depth map (n x camera_height x camera_width tensor)
        """
        with torch.no_grad():
            
            mask, pts_flat = self.get_valid_points(sq_poses, sq_params)
            # print(pts_flat.shape, sq_poses.shape, sq_params.shape)
            
            n_sq = len(sq_poses)
            n_total_pnts = pts_flat.shape[0] * pts_flat.shape[1]

            # 
            split_indices = mask.sum(dim=[1,2])
            valid_indices = torch.arange(n_sq).to(self.device).repeat_interleave(split_indices * (self.N_samples+2))
            valid_indices = torch.arange(n_sq).to(self.device).repeat(n_total_pnts, 1) == valid_indices.unsqueeze(-1)

            # calculate sdf values for sampled points
            sdf_values = sq_distance(
                pts_flat.reshape(-1, 3), 
                sq_poses,
                sq_params
            ) # n x n_sq --> n (mask.sum(dim=[1,2]).tolist())
            sdf_values = sdf_values[valid_indices]
    
            occ_values = torch.sigmoid(self.sharpness * (1 - sdf_values))
            
            outputs = occ_values.reshape(-1, self.N_samples+2)

            # visibility function
            visibility = torch.cumsum(outputs, dim=-1)
            visibility = torch.exp(-self.tau * visibility)
            valid_depth = self.integrate(visibility, pts_flat)

            depth = torch.zeros(n_sq, int(self.height//self.reduce_ratio), int(self.width//self.reduce_ratio)).to(self.device)
            depth[mask] = valid_depth
            depth[~mask] = self.far

        return depth

    def get_all_depths_old(self, sq_poses, sq_params):
        """
        input: sq_poses (n x 4 x 4 torch tensor)
               sq_params (n x 5 torch tensor)
        output: depth map (n x camera_height x camera_width tensor)
        """
        with torch.no_grad():
            
            depth_list = []
            mask, pts_flat = self.get_valid_points(sq_poses, sq_params)
            pts_flat_per_sq = pts_flat.split(mask.sum(dim=[1,2]).tolist()) 

            # calculate sdf values for sampled points
            for i in range(len(pts_flat_per_sq)):
                pts_flat = pts_flat_per_sq[i]
                sdf_values = sq_distance(
                    pts_flat.reshape(-1, 3), 
                    sq_poses[[i]],
                    sq_params[[i]]
                ).squeeze()
                
                occ_values = torch.sigmoid(self.sharpness * (1 - sdf_values))
                
                outputs = occ_values.reshape(-1, self.N_samples+2)

                # visibility function
                visibility = torch.cumsum(outputs, dim=-1)
                visibility = torch.exp(-self.tau * visibility)
                
                valid_depth = self.integrate(visibility, pts_flat)
                depth = torch.zeros(int(self.height//self.reduce_ratio), int(self.width//self.reduce_ratio)).to(self.device)
                
                
                depth[mask[i]] = valid_depth
                depth[~mask[i]] = self.far
                
                depth_list.append(depth)
            depth_list = torch.stack(depth_list)
        
        return depth_list

    def depth_render(self, sq_poses, sq_params):
        with torch.no_grad():
            depth_list = self.get_all_depths(sq_poses, sq_params)
            depth_map = depth_list.min(dim=0).values
        return depth_map
    
    def integrate(self, visibility, ray_pts):
        dt = ray_pts.diff(dim=-2).norm(dim=-1)
        cen_visibility = visibility[...,:-1] + visibility.diff() * 0.5
        depth_predicted = torch.sum(cen_visibility * dt, dim=-1) + self.near
        return depth_predicted
    
    def depth_render_batchwise(self, target_object_poses, target_object_sq_params, other_objects_poses, other_objects_sq_params, batch_size=100):
        """
        Note that other objects are not changed throughout the batch. Only target objects will be changed in the batch.
        
        input: target_object_poses (n x 4 x 4 torch tensor)
               target_object_sq_params (n x 5 torch tensor)
               other_objects_poses (m x x 4 x 4 torch tensor)
               other_objects_sq_params (m x 5 torch tensor)
        output: depth map (n x camera_height x camera_width tensor)
        """
        with torch.no_grad():
            other_depth_map = self.depth_render(other_objects_poses, other_objects_sq_params).unsqueeze(0) # H x W
            depth_map_list = []
            for target_objects_pose_batch, target_objects_sq_param_batch in zip(target_object_poses.split(batch_size), target_object_sq_params.split(batch_size)):
                target_depth_map_list = self.get_all_depths(target_objects_pose_batch, target_objects_sq_param_batch) # n x H x W
                depth_list = torch.cat(
                    [
                    target_depth_map_list.unsqueeze(1),
                    other_depth_map.repeat(len(target_objects_pose_batch), 1, 1, 1)
                    ], dim=1)
                depth_map = depth_list.min(dim=1).values
                depth_map_list.append(depth_map)
        return torch.cat(depth_map_list, dim=0)
