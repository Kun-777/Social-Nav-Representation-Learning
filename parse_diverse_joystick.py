import os
import argparse
import glob
import numpy as np
import pickle
import cv2
from termcolor import cprint
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R

def get_affine_mat(x, y, theta):
    """
    Returns the affine transformation matrix for the given parameters.
    """
    theta = np.deg2rad(theta)
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


def get_affine_matrix_quat(x, y, quaternion):
    theta = R.from_quat(quaternion).as_euler('XYZ')[2]
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing data with diverse joystick sequence')
    parser.add_argument('--data_path', type=str, default='/scratch/fulinj/social_nav/data/')
    parser.add_argument('--export_path', type=str, default='/scratch/fulinj/social_nav/joyseq_diverse/')
    parser.add_argument('--similarity_threshold', type=int, default=0.05)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise Exception('Data path does not exist')

    if not os.path.exists(args.export_path):
        raise Exception('Export path does not exist')

    pickle_file_paths = glob.glob(os.path.join(args.data_path, '*_data.pkl'))[:10]
    straight = np.asarray([1.6, 0, 0] * 150)
    for i, pickle_file_path in enumerate(tqdm(pickle_file_paths)):
        print('reading pickle file : ', pickle_file_path)
        if not os.path.exists(pickle_file_path.replace('_data.pkl', '_final.pkl')):
            raise Exception("Pickle file does not exist. Please process the pickle file first..")
        else:
            cprint('Pickle file exists. Loading from pickle file')
            data = pickle.load(open(pickle_file_path.replace('_data.pkl', '_final.pkl'), 'rb'))
        
        bev_lidar_dir: str = pickle_file_path[:-8] + 'final'
        delay_frame = 0
        for delay_i in range(len(data['joystick'])):
            if abs(data['joystick'][delay_i][0]) < 0.5:
                delay_frame = delay_i
            else:
                break
        cprint('Delay frame is : ' + str(delay_frame),'yellow', attrs=['bold'])

        for j, joy_seq in enumerate(data['future_joystick']):
            if j < delay_frame + 20:
                continue
            joy_flat = np.asarray(joy_seq[:150]).flatten()
            joy_dist = ((joy_flat - straight) ** 2).mean()
            # bev_lidar_image = np.array(Image.open(os.path.join(bev_lidar_dir, f'{j}.png')))
            # print(bev_lidar_image.shape)
            if joy_dist > args.similarity_threshold:
                # show trajectory and store image
                bev_lidar_image = np.array(Image.open(os.path.join(bev_lidar_dir, f'{j}.png')))
                print(bev_lidar_image.shape)
                bev_lidar_image = cv2.cvtColor(bev_lidar_image, cv2.COLOR_GRAY2BGR)
                T_odom_robot = get_affine_matrix_quat(data['odom'][-1][0], data['odom'][-1][1], data['odom'][-1][2])
                for goal in data['human_expert_odom'][-1][:200]:
                    T_odom_goal = get_affine_matrix_quat(goal[0], goal[1], goal[2])
                    T_robot_goal = np.matmul(
                        np.linalg.pinv(T_odom_robot), T_odom_goal)
                    T_c_f = [T_robot_goal[0, 2], T_robot_goal[1, 2]]
                    t_f_pixels = [int(T_c_f[0] / 0.05) + 200,
                                int(-T_c_f[1] / 0.05) + 200]
                    bev_lidar_image = cv2.circle(
                        bev_lidar_image, (t_f_pixels[0], t_f_pixels[1]), 1, (0, 0, 255), -1)
                # os.chdir(args.export_path)
                # cv2.imwrite(os.path.join(bev_lidar_dir, f'{j}.png'), bev_lidar_image)
                cv2.imshow('bev_lidar', bev_lidar_image)
                cv2.waitKey(1)
