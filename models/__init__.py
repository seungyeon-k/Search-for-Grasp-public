import torch
import os 
from omegaconf import OmegaConf
from utils.utils import dictconfig_to_dict
from .cVAE import CondVAE
from .dgcnn import DGCNN
from models.modules import (
    FC_vec,
)

from models.motion_prediction_network import (
    MotionPredictionNetwork
)
from models.pushing_dynamics import (
    PushingDynamics,
    EquivariantPushingDynamics
)
from models.sqnet import SuperquadricNetwork
from models.segmenation_network import SegmentationNetwork

def get_model(model_cfg, *args, **kwargs):
    name = model_cfg["arch"]

    if model_cfg.get('backbone', False):
        cfg_backbone = model_cfg.pop('backbone')
        backbone = get_backbone_instance(cfg_backbone['arch'])(**cfg_backbone)

        model = _get_model_instance(name)
        model = model(backbone, **model_cfg, **kwargs)
    else:
        model = _get_model_instance(name)
        model = model(**model_cfg, **kwargs)        
        
    return model

def _get_model_instance(name):
    try:
        return {
            'sqnet': SuperquadricNetwork,
            'segnet': SegmentationNetwork,
            'pushingdynamics': get_pushing_dynamics,
            'equi_pushingdynamics': get_pushing_dynamics,
        }[name]
    except:
        raise ("Model {} not available".format(name))

def get_backbone_instance(name):
    try:
        return {
            'dgcnn': DGCNN,
        }[name]
    except:
        raise (f"Backbone {name} not available")

def get_pushing_dynamics(**model_cfg):
    if model_cfg["arch"] == "pushingdynamics":
        module = get_net(**model_cfg["module"])
        regressor = PushingDynamics(module)
    elif model_cfg["arch"] == "equi_pushingdynamics":
        module = get_net(**model_cfg["module"])
        plug_in_type = model_cfg["plug_in_type"]
        regressor = EquivariantPushingDynamics(module, plug_in_type)
    return regressor

def get_net(**kwargs):
    if kwargs["arch"] == "sqpdnet":
        net = MotionPredictionNetwork(**kwargs)
    return net

def load_pretrained(identifier, config_file, ckpt_file, root='pretrained', **kwargs):
    """
    load pre-trained model.
    identifier: '<model name>/<run name>'. e.g. 'ae_mnist/z16'
    config_file: name of a config file. e.g. 'ae.yml'
    ckpt_file: name of a model checkpoint file. e.g. 'model_best.pth'
    root: path to pretrained directory
    """
    config_path = os.path.join(root, identifier, config_file)
    ckpt_path = os.path.join(root, identifier, ckpt_file)
    cfg = OmegaConf.load(config_path)

    model = get_model(cfg.model)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'model_state' in ckpt:
        ckpt = ckpt['model_state']
    model.load_state_dict(ckpt)
    
    return model, cfg