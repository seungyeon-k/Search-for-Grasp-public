import argparse
from omegaconf import OmegaConf
from control.controller import Controller
import pybullet as p 


def run(cfg):
    
    # setup device
    device = cfg.device
    epoch = cfg.epoch

    # control
    exp_idx = 0
    while(exp_idx < epoch):
        # setup controller
        controller = Controller(
        cfg.controller, 
        device=device, 
        )
        
        success_to_init = controller.spawn_datas(exp_idx=exp_idx)
        p.disconnect()
        
        if success_to_init:
            exp_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', default=0)

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.device == 'cpu':
        cfg.device = 'cpu'
    else:
        cfg.device = f'cuda:{args.device}'

    run(cfg)