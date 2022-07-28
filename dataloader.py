import cv2
import numpy as np
import os
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from termcolor import cprint
import pytorch_lightning as pl

class BTSNDataset(Dataset):
    """
    Create a data set object from a single pkl file. Each index of this dataset has a 5 stacked lidar images, a flattened 1D array of joystick commands from current position to the goal 10m into the future, and a flattened 1D array of trajectory from current position to the goal 10m into the future.
    E.g. dataset[0][0] is the 5 stacked lidar images at t=0 (after delay), dataset[0][1] is the joystick commands in 10m into the future at t=0, dataset[0][2] is the trajectory in 10m into the future at t=0. 
    """

    def __init__(self, pickle_file_path, delay_frame=30):
        # load the pickle file
        if not os.path.exists(pickle_file_path.replace('_data.pkl', '_final.pkl')):
            raise Exception(
                "Pickle file does not exist. Please process the pickle file first..")
        else:
            cprint('Pickle file exists. Loading from pickle file')
            self.data = pickle.load(
                open(pickle_file_path.replace('_data.pkl', '_final.pkl'), 'rb'))
        
        # get path to bev_lidar_images
        # but do not load images into memory
        self.bev_lidar_dir: str = pickle_file_path[:-4]

        # skip the first few data points - delay in movement after pressing record button
        self.delay_frame = delay_frame


        for delay_i in range(len(self.data['joystick'])):
            if abs(self.data['joystick'][delay_i][0]) < 0.5:
                self.delay_frame = delay_i
            else:
                break
        cprint('Delay frame is : ' + str(self.delay_frame),'yellow', attrs=['bold'])


    def __len__(self):
        return len(self.data['local_goals']) - self.delay_frame - 20
    
    def __getitem__(self, idx):
        idx = self.delay_frame + idx

        bev_img_stack = [np.array(Image.open(os.path.join(
            self.bev_lidar_dir, f'{x}.png'))) for x in range(idx, idx + 20)]
        bev_img_stack_odoms = self.data['odom'][idx:idx+20]
        bev_img_stack = [bev_img_stack[i] for i in [0, 5, 10, 15, 19]]
        bev_img_stack_odoms = [bev_img_stack_odoms[i]
                               for i in [0, 5, 10, 15, 19]]
        
        # rotate previous frames to current frame
        last_frame = bev_img_stack_odoms[-1]
        T_odom_5 = self.get_affine_matrix_quat(
            last_frame[0], last_frame[1], last_frame[2])
        T_odom_4 = self.get_affine_matrix_quat(bev_img_stack_odoms[-2][0],
                                               bev_img_stack_odoms[-2][1],
                                               bev_img_stack_odoms[-2][2])
        T_4_5 = self.affineinverse(T_odom_4) @ T_odom_5
        T_odom_3 = self.get_affine_matrix_quat(bev_img_stack_odoms[-3][0],
                                               bev_img_stack_odoms[-3][1],
                                               bev_img_stack_odoms[-3][2])
        T_3_5 = self.affineinverse(T_odom_3) @ T_odom_5
        T_odom_2 = self.get_affine_matrix_quat(bev_img_stack_odoms[-4][0],
                                               bev_img_stack_odoms[-4][1],
                                               bev_img_stack_odoms[-4][2])
        T_2_5 = self.affineinverse(T_odom_2) @ T_odom_5
        T_odom_1 = self.get_affine_matrix_quat(bev_img_stack_odoms[-5][0],
                                               bev_img_stack_odoms[-5][1],
                                               bev_img_stack_odoms[-5][2])
        T_1_5 = self.affineinverse(T_odom_1) @ T_odom_5
        # now do the rotations
        T_1_5[:, -1] *= -20
        T_2_5[:, -1] *= -20
        T_3_5[:, -1] *= -20
        T_4_5[:, -1] *= -20
        bev_img_stack[0] = cv2.warpAffine(
            bev_img_stack[0], T_1_5[:2, :], (401, 401))
        bev_img_stack[1] = cv2.warpAffine(
            bev_img_stack[1], T_2_5[:2, :], (401, 401))
        bev_img_stack[2] = cv2.warpAffine(
            bev_img_stack[2], T_3_5[:2, :], (401, 401))
        bev_img_stack[3] = cv2.warpAffine(
            bev_img_stack[3], T_4_5[:2, :], (401, 401))

        # cut image to 400 x 400
        new_img_stack = []
        for bev_img in bev_img_stack:
            new_img = []
            for i, row in enumerate(bev_img):
                row = np.resize(row, 400)
                if i != 400:
                    new_img.append(row)
            new_img = np.asarray(new_img)
            new_img_stack.append(new_img)


        # combine the 5 single-channel images into a single image of 5 channels
        bev_img_stack = np.asarray(new_img_stack).astype(np.float32)
        # bev_img_stack = np.asarray(bev_img_stack).astype(np.float32)
        bev_img_stack = bev_img_stack / 255.0  # normalize the image

        # truncate to 300 future joystick commands
        future_joystick_values = np.asarray(self.data['future_joystick'][idx+20-1][:300]).flatten()

        # flatten the trajectory information in each entry, [[x, y, [a, b, c, d]], ...] to [[x, y, a, b, c, d], ...] 6*n

        flattened_traj = []
        for trajectory_val in self.data['human_expert_odom'][idx+20-1][:100]:
            # temp = []
            for single_traj in trajectory_val:
                if type(single_traj) is list:
                    for orientation in single_traj:
                        # temp.append(orientation)
                        flattened_traj.append(orientation)
                else:
                    # temp.append(single_traj)
                    flattened_traj.append(single_traj)
            # flattened_traj.append(temp)

        future_trajectory = np.asarray(flattened_traj)

        return bev_img_stack, future_joystick_values, future_trajectory

    @staticmethod
    def get_affine_mat(x, y, theta):
        """
                Returns the affine transformation matrix for the given parameters.
                """
        theta = np.deg2rad(theta)
        return np.array([[np.cos(theta), -np.sin(theta), x],
                         [np.sin(theta), np.cos(theta), y],
                         [0, 0, 1]])

    @staticmethod
    def get_affine_matrix_quat(x, y, quaternion):
        theta = R.from_quat(quaternion).as_euler('XYZ')[2]
        return np.array([[np.cos(theta), -np.sin(theta), x],
                         [np.sin(theta), np.cos(theta), y],
                         [0, 0, 1]])

    @staticmethod
    def affineinverse(M):
        tmp = np.hstack((M[:2, :2].T, -M[:2, :2].T @ M[:2, 2].reshape((2, 1))))
        return np.vstack((tmp, np.array([0, 0, 1])))



class MyDataLoader(pl.LightningDataModule):
    '''
    Load all .pkl files and lidar scans in data_path. Create a BTSNDataset for each .pkl file, and concat all into one. 
    '''
    def __init__(self, data_path, batch_size=32, num_workers=0, delay_frame=30):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        # load the pickle files
        pickle_file_paths = glob.glob(os.path.join(data_path, '*_data.pkl'))

        self.validation_dataset, self.training_dataset = [], []
        self.training_sample_weights = []

        self.val_pickle_files, self.train_pickle_files = [], []

        for i, pickle_file_path in enumerate(tqdm(pickle_file_paths)):
            print('reading pickle file : ', pickle_file_path)
            if i < int(len(pickle_file_paths)*0.75):
                self.train_pickle_files.append(pickle_file_path)
                tmp = BTSNDataset(pickle_file_path, delay_frame=delay_frame)
                # skip if the specific rosbag was small and wasn't long enough
                if len(tmp) <= 0:
                    continue
                self.training_dataset.append(tmp)
            else:
                self.val_pickle_files.append(pickle_file_path)
                tmp = BTSNDataset(pickle_file_path, delay_frame=delay_frame)
                if len(tmp) <= 0:
                    continue
                self.validation_dataset.append(tmp)
        
        self.training_dataset, self.validation_dataset = ConcatDataset(self.training_dataset), ConcatDataset(self.validation_dataset)

        # save the pickle file path for train and validation in the same directory
        cprint('Saving val and train dataset names in text files...',
               'yellow', attrs=['bold', 'blink'])
        with open(os.path.join(data_path, 'train_pickle_files.txt'), 'w') as f:
            for pickle_file_path in self.val_pickle_files:
                f.write(pickle_file_path + '\n')
        with open(os.path.join(data_path, 'val_pickle_files.txt'), 'w') as f:
            for pickle_file_path in self.val_pickle_files:
                f.write(pickle_file_path + '\n')
        cprint('Dumped val and train dataset names in text files...',
               'green', attrs=['bold', 'blink'])

        cprint('Loaded Datasets !!', 'green', attrs=['bold'])
        cprint('Num training datapoints : ' +
               str(len(self.training_dataset)), 'green', attrs=['bold'])
        cprint('Num validation datapoints : ' +
               str(len(self.validation_dataset)), 'green', attrs=['bold'])
        
    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=not (len(self.training_dataset) % self.batch_size == 0.0))

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,
                        drop_last=not (len(self.validation_dataset) % self.batch_size == 0.0))