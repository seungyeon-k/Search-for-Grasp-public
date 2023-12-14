import os.path as osp
from tqdm import tqdm
import argparse
import h5py
from datetime import datetime

from data_generation.sim_with_shelf_env import SimulationShelfEnv
from data_generation.utils import mkdir, debugging_windows

def main(args):

    # load args
    for key in vars(args):
        print(f"[{key}] = {getattr(args, key)}")

    # dataset folder name
    if args.folder_name is None:
        folder_name = datetime.now().strftime('%Y%m%d-%H%M')
    else:
        folder_name = args.folder_name
    data_path = osp.join(args.data_path, folder_name)
    mkdir(data_path, clean=False)

    # declare simulation env
    env = SimulationShelfEnv(enable_gui=args.enable_gui)

    # save camera parameters
    f = h5py.File(osp.join(data_path, 'camera_param.hdf5'), 'w')
    camera_param = env.sim.camera_params[0]
    for key in camera_param.keys():
        f.create_dataset(key, data=camera_param[key]) 
    f.close()

    # data generation
    for split in ['training', 'validation', 'test']:
        if getattr(args, split+"_num") == 0:
            continue
        else:
            dataset_len = getattr(args, split+"_num")

        # split folder name
        split_data_path = osp.join(data_path, split)
        mkdir(split_data_path, clean=False)

        # known/random objects
        if args.knowledge_random:
            knowledge = 'random'
        else:
            knowledge = 'known'
        print(knowledge)

        # initialize
        rollout = 0
        data_num = 0
        flag = False
        pbar = tqdm(
            total=dataset_len, 
            desc=f"{split} dataset ... ", 
            leave=False
        )

        while True:
            # reset simulation
            env.reset(
                args.object_types, 
                knowledge, 
                args.num_objects, 
                enable_stacking=args.push_3d
            )

            # get data generation setting
            only_recognition = args.only_recognition
            if not only_recognition:
                total_push_num = args.push_num
            else:
                total_push_num = 1

            for push_num in range(total_push_num):
                # poke simulation
                output = env.poke(only_recognition=only_recognition)
                if output is None:
                    break

                # make hdf5 file
                file_name = f"{rollout:04}_{push_num:02}.hdf5"
                f = h5py.File(osp.join(split_data_path, file_name), 'w')

                # save output
                if not only_recognition:
                    save_key_list = [
                        'pc_down_wo_plane', 
                        'labels_down_wo_plane',
                        'pc_gt', 
                        'object_info', 
                        'positions_old', 
                        'orientations_old', 
                        'action_coord', 
                        'positions_new', 
                        'orientations_new',
                        'action_pixel', 
                        'tsdf', 
                        'mask_3d_old', 
                        'scene_flow_3d',
                        'organized_pc', 
                        'scene_flow_2d', 
                        'mask_2d_old',
                        'mask_3d_new', 
                        'mask_2d_new', 
                        'depth_image_old', 
                        'depth_image_new',
                    ]
                else:
                    save_key_list = [
                        'pc_down_wo_plane', 
                        'labels_down_wo_plane',
                        'pc_gt', 
                        'object_info', 
                        'positions_old', 
                        'orientations_old', 
                        # 'tsdf', 
                        # 'mask_3d_old', 
                        # 'organized_pc', 
                        # 'mask_2d_old',
                        # 'depth_image_old', 
                    ]

                for key in set(save_key_list):
                    if type(output[key]) == str:
                        f.create_dataset(key, data=output[key]) 
                    else:
                        f.create_dataset(
                            key, 
                            data=output[key], 
                            compression='gzip', 
                            compression_opts=4
                        )

                # debug data
                if args.debug:
                    debugging_windows(output)

                # finish
                f.close()
                data_num += 1
                pbar.update(1)

                if data_num == dataset_len:
                    flag = True
                    break

            # roll out
            rollout += 1
            if flag:
                break

        pbar.close()

if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_gui', action='store_true', help='show PyBullet simulation window')
    parser.add_argument('--data_path', type=str, default='datasets', help="path to data")
    parser.add_argument('--folder_name', default=None, help="dataset folder name")
    parser.add_argument('--training_num', type=int, default=1, help="number of training scenes")
    parser.add_argument('--validation_num', type=int, default=1, help="number of validation scenes")
    parser.add_argument('--test_num', type=int, default=1, help="number of testing scenes")
    parser.add_argument('--object_types', nargs='+', default=['box', 'cylinder'], help="choose object types from 'box', 'cylinder'")
    parser.add_argument('--num_objects', type=int, default=4, help="number of objects")
    parser.add_argument('--known', action='store_true', help="unknown objects for validation and test")
    parser.add_argument('--push_3d', action='store_true', help="enable object stacking and falling")
    parser.add_argument('--push_num', type=int, default=20, help="number of pushing actions in each sequence")
    parser.add_argument('--debug', action='store_true', help="debugging point cloud")
    parser.add_argument('--only_recognition', action='store_true', help="no pushing and obtain only recognition data")
    parser.add_argument('--knowledge_random', action='store_true', help="debugging point cloud")
    args = parser.parse_args()
    
    main(args)