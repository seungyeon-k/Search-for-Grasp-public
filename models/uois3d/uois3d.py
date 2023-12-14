import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU
import yaml
from omegaconf import OmegaConf
import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2


from . import data_augmentation
from . import segmentation
from . import utilities as util_



def load_pretrained_model(model_cfg_dir):

    cfg = OmegaConf.load(model_cfg_dir)

    dsn_config = cfg["dsn_config"]
    rrn_config = cfg["rrn_config"]
    uois3d_config = cfg["uois3d_config"]
    checkpoint_dir = cfg["checkpoint_dir"]
    checkpoint_dir = os.path.join(os.path.abspath('.'), checkpoint_dir)

    dsn_filename = checkpoint_dir + cfg["dsn_filename"]
    rrn_filename = checkpoint_dir + cfg["rrn_filename"]
    
    uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                        dsn_filename,
                                        dsn_config,
                                        rrn_filename,
                                        rrn_config
                                        )
    
    
    return uois_net_3d

def preprocess_image(rgb_img, xyz_img):

    rgb_img = np.expand_dims(rgb_img, 0)
    rgb_img = data_augmentation.standardize_image(rgb_img)
    xyz_img = np.expand_dims(xyz_img, 0)
    xyz_img[np.isnan(xyz_img)] = 0
    batch = {
        'rgb' : data_augmentation.array_to_tensor(rgb_img),
        'xyz' : data_augmentation.array_to_tensor(xyz_img),
    }
    
    return batch
    

def make_figure(batch, fg_masks, center_offsets, initial_masks, seg_masks):
    
    # Get results in numpy
    seg_masks = np.squeeze(seg_masks.cpu().numpy(), 0)
    fg_masks = np.squeeze(fg_masks.cpu().numpy(), 0)
    center_offsets = np.squeeze(center_offsets.cpu().numpy().transpose(0,2,3,1), 0)
    center_offsets = np.linalg.norm(center_offsets,axis=2)
    initial_masks = np.squeeze(initial_masks.cpu().numpy(), 0)   
    rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
    xyz_imgs = util_.torch_to_numpy(batch['xyz'].cpu(), is_standardized_image=True)
        
    num_objs_seg = np.unique(seg_masks).max() + 1
    num_objs_init = np.unique(initial_masks).max() + 1
    rgb = rgb_imgs[0].astype(np.uint8)
    depth = xyz_imgs[0,...,2]
    fg_masks_plot = util_.get_color_mask(fg_masks, nc=num_objs_init)
    center_offsets_plot = center_offsets
    initial_masks_plot = util_.get_color_mask(initial_masks, nc=num_objs_init)
    seg_mask_plot = util_.get_color_mask(seg_masks, nc=num_objs_seg)
    images = [rgb, depth, fg_masks_plot, center_offsets_plot, initial_masks_plot, seg_mask_plot]
    titles = [f'Image', 'Depth',
            f"Foreground Masks.",
            f"Center Offsets.",
            f"Initial Masks. #objects: {num_objs_init}",
            f"Refined Masks. #objects: {num_objs_seg}",
            ]
    
    # return util_.subplotter_numpy(images, titles, fig_num=1)
    return util_.subplotter(images, titles, fig_num=1)