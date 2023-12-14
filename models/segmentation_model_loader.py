import numpy as np
import torch
import matplotlib.pyplot as plt

class SegmentationModel:
    def __init__(self, model_name, model_cfg_dir, load_pretrained_weights):
        self.model_name = model_name
        
        if model_name == "uois3d":
            from .uois3d.uois3d import load_pretrained_model, preprocess_image, make_figure
            self.model = load_pretrained_model(model_cfg_dir)
            self.preprocess_image = preprocess_image
            self.make_figure = make_figure 
    
    def preprocess(self, rgb_img, xyz_img):
        if self.model_name == "uois3d":
            return self.preprocess_image(rgb_img, xyz_img)
            
    def forward(self, x):
        if self.model_name == "uois3d":
            return self.model.run_on_batch(x)
    
    def draw_figures(self, x, y, figure_name = False):
        if self.model_name == "uois3d":
            fig_numpy = self.make_figure(x, *y)
            if figure_name is not False:
                plt.savefig(figure_name)
            return fig_numpy
    
def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # shape: [H X W X 3]
    return xyz_img / 1000