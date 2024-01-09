# Search-for-Grasp
The official repository for \<Leveraging 3D Reconstruction for Mechanical Search on Cluttered Shelves\> (Seungyeon Kim*, Young Hun Kim*, Yonghyeon Lee, and Frank C. Park, CoRL 2023).

> Finding and grasping a target object on a cluttered shelf, especially when the target is occluded by other unknown objects and initially invisible, remains a significant challenge in robotic manipulation. In this paper, we introduce a novel framework for finding and grasping the target object using a standard gripper, employing pushing and pick and-place actions. To achieve this, we introduce two indicator functions: (i) an existence function, determining the potential presence of the target, and (ii) a graspability function, assessing the feasibility of grasping the identified target. The core component of our approach involves leveraging a 3D recognition model, enabling efficient estimation of the proposed indicator functions and their associated dynamics models.

- *[Paper](https://proceedings.mlr.press/v229/kim23a/kim23a.pdf)* 
- *[Supplementary video](https://www.youtube.com/watch?v=FoejNGHf1XM&t=2s)*
- *[Slides](https://drive.google.com/file/d/12B6Xd9QTmh-tj19Ddhc-OTpQF9nHDPcd/view?usp=drive_link)*
- *[Poster](https://drive.google.com/file/d/125g_tewVmJ0L1F-pEZBpRGNSHcpZs8Kw/view?usp=drive_link)*
- *[Openreview](https://openreview.net/forum?id=ycy47ZX0Oc)*

## Preview
### Mechanical Search on Cluttered Shelves
![sim_results](figures/sim_results.PNG)
<I>Figure 1: An example trajectory of simulation manipulation. Each column shows the camera input
and action selection at each time step. In the simulation, surrounding objects are blue and the target
object is red. </I>

## Requirements
### Environment
The project is developed under a standard PyTorch environment.
- python 3.9
- pybullet 3.2.3
- pytorch
- matplotlib
- Open3D
- opencv-python
- dominate
- h5py
- tensorboardx (optional)

<!-- - tqdm
- h5py
- Open3D
- scipy
- scikit-learn 
- opencv-python
- imageio
- matplotlib
- scikit-image
- dominate
- numba -->

### Pretrained model
Pre-trained models should be stored in `pretrained/`. The pre-trained models are already provided in this repository. After set up, the `pretrained/` directory should be follows.
```
pretrained
├── segmentation_config
│   └── pretrained
│       ├── segmentation_config.yml
│       └── model_best.pkl
├── sqpdnet_config
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
python control.py --config configs/control_config.yml
```

### Spawn Object Sets for Experiments
The scripts in Pybullet simulator are as follows:
```
python spawn.py
```

### Replay Episodes
The scripts in Pybullet simulator are as follows:
```
python replay_episode.py 
```
pybullet unrecognized option 'preset'
conda update ffmpeg


### (Optional) Train Models


If you want to generate your own custom dataset, run the following script:
```shell
python data_generation.py --enable_gui                # PyBullet UI on/off
                          --folder_name test          # folder name of the generated dataset
                          --object_types box cylinder # used object types for data generation
                          --num_objects 4             # can be 1~4; currently the max number of object is 4
                          --push_num 20               # max number of pushing per sequence
                          --training_num 150          # the number of training set; total number of training set is (training_num * push_num)
                          --validation_num 15         # the number of validation set; total number of validation set is (validation_num * push_num)
                          --test_num 15               # the number of test set; total number of test set is (test_num * push_num)
``` 

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