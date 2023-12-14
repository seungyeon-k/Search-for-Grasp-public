# Search-for-Grasp
The official repository for \<Leveraging 3D Reconstruction for Mechanical Search on Cluttered Shelves\> (Seungyeon Kim*, Young Hun Kim*, Yonghyeon Lee, and Frank C. Park, CoRL 2023).

> Finding and grasping a target object on a cluttered shelf, especially when the target is occluded by other unknown objects and initially invisible, remains a significant challenge in robotic manipulation. In this paper, we introduce a novel framework for finding and grasping the target object using a standard gripper, employing pushing and pick and-place actions. To achieve this, we introduce two indicator functions: (i) an existence function, determining the potential presence of the target, and (ii) a graspability function, assessing the feasibility of grasping the identified target. The core component of our approach involves leveraging a 3D recognition model, enabling efficient estimation of the proposed indicator functions and their associated dynamics models.

- *[Paper](https://proceedings.mlr.press/v229/kim23a/kim23a.pdf)* 
- *[Supplementary video](https://www.youtube.com/watch?v=FoejNGHf1XM&t=2s)*
- *[Slides](https://drive.google.com/file/d/12B6Xd9QTmh-tj19Ddhc-OTpQF9nHDPcd/view?usp=drive_link)*
- *[Poster](https://drive.google.com/file/d/125g_tewVmJ0L1F-pEZBpRGNSHcpZs8Kw/view?usp=drive_link)*
- *[Openreview](https://openreview.net/forum?id=ycy47ZX0Oc)*

The complete code will be available no later than December 21th. Thank you for waiting!

<!-- ## Preview
### Mechanical Search on Cluttered Shelves
In progress...

## Requirements
### Environment
The project is developed under a standard PyTorch environment.
- python 3.9
- pybullet 3.2.3
- pytorch
- tensorboardx
- tqdm
- h5py
- Open3D
- scipy
- scikit-learn 
- opencv-python
- imageio
- matplotlib
- scikit-image
- dominate
- numba

### Pretrained model
Pre-trained models should be stored in `pretrained/`. The pre-trained models are already provided in this repository. After set up, the `pretrained/` directory should be follows.
```
pretrained
├── segmentation_config
│   └── pretrained
│       ├── segmentation_config.yml
│       └── model_best.pkl
├── sqpdnet_2d_motion_only_config
│   └── pretrained
│       ├── sqpdnet_2d_motion_only_config.yml
│       └── model_best.pkl
└── recognition_config
    └── pretrained
        ├── recognition_config.yml
        └── model_best.pkl
```

## Running
### Control in Simulation Environment
The control scripts in Pybullet simulator are as follows:
```
python control.py --config configs/control_sim/control_sim_{X}_config.yml
```
- `X` is either `moving`, `singulation`, `grasping_clutter`, `grasping_large`, or `moving_interactive`. -->





## Citation
If you found this repository useful in your research, please consider citing:
```
@inproceedings{kim2023leveraging,
  title={Leveraging 3D Reconstruction for Mechanical Search on Cluttered Shelves},
  author={Kim, Seungyeon and Kim, Young Hun and Lee, Yonghyeon and Park, Frank C},
  booktitle={Conference on Robot Learning},
  pages={822--848},
  year={2023},
  organization={PMLR}
}
```