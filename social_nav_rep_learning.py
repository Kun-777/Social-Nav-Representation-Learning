import cv2
import numpy as np
import torch
import argparse
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from datetime import datetime
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
from termcolor import cprint
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import tensorboard as tb
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from PIL import Image
from barlow_twins_loss import BarlowTwinsLoss
from vicreg import VICReg
from vicreg import W_VICReg
from vicreg import M_VICReg
from distributed import init_distributed_mode
from skimage.measure import block_reduce
from itertools import zip_longest
from vitNetwork import vitNetwork
from cnnNetwork import CNNNetwork


class SNModel(pl.LightningModule):
    def __init__(self, lambd, lr, weight_decay, patch_size, use_vit):
        super(SNModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patch_size = patch_size
        self.use_vit = use_vit
        if use_vit:
            self.model = vitNetwork(joystick_input_size=300, img_size=240, img_channels=5, patch_size=patch_size)
        else:
            self.model = CNNNetwork(joystick_input_size=300)
        self.model.to(self.device)
        self.barlow_twins_loss = BarlowTwinsLoss(lambd)
        self.vicreg_loss = VICReg()
        # self.w_vicreg_loss = W_VICReg()
        # self.m_vicreg_loss = M_VICReg()

    def forward(self, img_batch, joystick_batch, goal_batch):
        return self.model(img_batch, joystick_batch, goal_batch)
    
    def training_step(self, batch, batch_idx):
        img_batch, joystick_batch, goal_batch, weights = batch
        img_rep, joystick_rep = self.forward(img_batch.float(), joystick_batch.float(), goal_batch.float())
        # loss = self.barlow_twins_loss(img_rep, joystick_rep)
        loss = self.vicreg_loss(img_rep, joystick_rep)
        # loss = self.w_vicreg_loss(img_rep, joystick_rep, weights)
        # loss = self.m_vicreg_loss(img_rep, joystick_rep, masks)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_batch, joystick_batch, goal_batch, weights = batch
        img_rep, joystick_rep = self.forward(img_batch.float(), joystick_batch.float(), goal_batch.float())
        # loss = self.barlow_twins_loss(img_rep, joystick_rep)
        loss = self.vicreg_loss(img_rep, joystick_rep)
        # loss = self.w_vicreg_loss(img_rep, joystick_rep, weights)
        # loss = self.m_vicreg_loss(img_rep, joystick_rep, masks)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        if self.use_vit:
            if self.current_epoch % 5 == 0 and batch_idx % 100 == 0:
                img_batch, _, goal_batch = batch
                with torch.no_grad():
                    attentions = self.model.get_last_selfattention(img_batch=img_batch.float().cuda(), goal_batch=goal_batch.float().cuda())
                    nh = attentions.shape[1] # number of head
                    # we keep only the output patch attention
                    numfeat = img_batch.shape[-1] // self.patch_size
                    attentions = attentions[63, :, 0, 1:-1].reshape(nh, -1)
                    attentions = attentions.reshape(nh, 240 // self.patch_size, 240 // self.patch_size)
                    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0].cpu().detach().numpy()
                    orig_img = img_batch[63][0].cpu().detach().numpy()
                    for i in range(5):
                        orig_img = cv2.addWeighted(orig_img,1,img_batch[63][i].cpu().detach().numpy(),np.power(0.85,i*4),0)
                    self.logger.experiment.add_image('original img: {}_{}'.format(batch_idx, dataloader_idx), torch.from_numpy(orig_img).unsqueeze(0), self.current_epoch)
                    self.logger.experiment.add_image('attention 0: {}_{}'.format(batch_idx, dataloader_idx), torch.from_numpy(
                        attentions[0]*255).unsqueeze(0), self.current_epoch)
        

    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

class SNDataset(Dataset):
    """
    Create a data set object from a single pkl file. Each index of this dataset has a 5 stacked lidar images, a flattened 1D array of joystick commands from current position to the goal 10m into the future, and a flattened 1D array of trajectory from current position to the goal 10m into the future.
    E.g. dataset[0][0] is the 5 stacked lidar images at t=0 (after delay), dataset[0][1] is the joystick commands in 10m into the future at t=0, dataset[0][2] is the trajectory in 10m into the future at t=0. 
    """

    def __init__(self, pickle_file_path, delay_frame=30):
        super(SNDataset, self).__init__()
        # load the pickle file
        if not os.path.exists(pickle_file_path.replace('_data.pkl', '_final.pkl')):
            raise Exception(
                "Pickle file does not exist. Please process the pickle file first..")
        else:
            cprint('Pickle file exists. Loading from pickle file')
            self.data = pickle.load(
                open(pickle_file_path.replace('_data.pkl', '_final.pkl'), 'rb'))

        straight = np.asarray([1.6, 0, 0] * 150)
        self.masks = [1 for _ in self.data['future_joystick']]
        for i, joy_seq in enumerate(self.data['future_joystick']):
            joy_flat = np.asarray(joy_seq[:150]).flatten()
            joy_dist = ((joy_flat - straight) ** 2).mean()
            if joy_dist < 0.005:
                self.masks[i] = 0
        # self.data['future_joystick'] = self.data['future_joystick'][masks == 1]
        
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
        # return len(self.data['future_joystick']) - self.delay_frame - 20
        return self.masks[self.delay_frame + 20 - 1:].count(1)
    
    def __getitem__(self, idx):
        # idx = self.delay_frame + idx
        # find #idx non-zero entry in masks
        for i, mask in enumerate(self.masks):
            # skip the delay frames and the first 19
            if i < self.delay_frame + 19:
                continue
            if mask == 1:
                if idx == 0:
                    idx = i - 19
                    break
                idx -= 1
                    

        bev_img_stack = [np.array(Image.open(os.path.join(
            self.bev_lidar_dir, f'{x}.png'))) for x in range(idx + 1, idx + 21)]
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

        # combine the 5 single-channel images into a single image of 5 channels
        bev_img_stack = np.asarray(bev_img_stack).astype(np.float32)
        bev_img_stack = bev_img_stack / 255.0  # normalize the image

        # cut image to 240 x 240
        # bev_img_stack = bev_img_stack[:, 80:320, 80:320]

        # represent image by mean and variance of each 4x4 sub image
        # means = block_reduce(bev_img_stack, block_size=(1, 4, 4), func=np.mean)
        # vars = block_reduce(bev_img_stack, block_size=(1, 4, 4), func=np.var)
        # bev_img_stack = np.concatenate((means, vars), axis=0)

        # truncate to 150 future joystick commands
        # future_joystick_values = np.asarray(self.data['future_joystick'][idx+20-1][:150]).flatten()
        future_joystick_values = np.asarray(flatten(self.data['human_expert_odom'][idx+20-1][:50])).flatten()

        # fixed_joystick_values = np.asarray(self.data['future_joystick'][idx+20-1][:150]).flatten()
        # straight = np.asarray([1.6, 0, 0] * 150)
        # joy_dist = ((fixed_joystick_values - straight) ** 2).mean()
        # mask = 1
        # if joy_dist < 0.005:
            # mask = 0

        # future_trajectory = np.asarray(flattened_traj)

        relative_goal_pos = np.asarray([self.data['local_goal_human_odom'][idx+20-1][-1][0],self.data['local_goal_human_odom'][idx+20-1][-1][1]])

        var_weight = (np.asarray(self.data['future_joystick'][idx+20-1][:150]).flatten()).var()

        return bev_img_stack, future_joystick_values, relative_goal_pos, var_weight

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
    Load all .pkl files and lidar scans in data_path. Create a SNDataset for each .pkl file, and concat all into one. 
    '''
    def __init__(self, data_path, batch_size=32, num_workers=0, delay_frame=30):
        super(MyDataLoader, self).__init__()
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
                tmp = SNDataset(pickle_file_path, delay_frame=delay_frame)
                # skip if the specific rosbag was small and wasn't long enough
                if len(tmp) <= 0:
                    continue
                self.training_dataset.append(tmp)
            else:
                self.val_pickle_files.append(pickle_file_path)
                tmp = SNDataset(pickle_file_path, delay_frame=delay_frame)
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

def flatten(nested_list):
   return list(zip( * _flattengen(nested_list)))

def _flattengen(iterable):
    for element in zip_longest( * iterable, fillvalue = ""):
        if isinstance(element[0], list):
            for e in _flattengen(element):
                yield e
        else :
            yield element

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn representations from SCAND dataset using Barlow Twins\' model.')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUS to use')
    parser.add_argument('--delay_frame', type=int, default=0, help='Number of initial frames to skip')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers to use')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Multiplier for batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lambd', type=float, default=0.0051, metavar='L', help='weight on off-diagonal terms for Barlow Twins loss')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--data_path', type=str, default='/scratch/fulinj/social_nav/data/')
    # parser.add_argument('--data_path', type=str, default='./test/')
    parser.add_argument('--notes', type=str, default='', help='notes for this specific run')
    parser.add_argument('--checkpoint', type=str, default='None')
    parser.add_argument('--use_pretrained_weight', type=str, default="")

    # distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    args = parser.parse_args()

    init_distributed_mode(args)

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise Exception('Data path does not exist')

    # check if checkpoint path exists
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # create model
    model = SNModel(lambd=args.lambd, lr=args.lr, weight_decay=args.weight_decay, patch_size=8, use_vit=False)

    if args.use_pretrained_weight == "":
        cprint('Not using pretrained weight', 'red', attrs=['bold'])
    else:
        # check if the file exists first
        if not os.path.exists(args.use_pretrained_weight):
            raise Exception('Pretrained weight file does not exist')

        # load pretrained weight from the path
        model.load_from_checkpoint(args.use_pretrained_weight)
        cprint('Loaded file from checkpoint : ' +
               str(args.use_pretrained_weight), 'green', attrs=['bold'])


    # load data
    dm = MyDataLoader(data_path=args.data_path, batch_size=args.batch_size, num_workers=args.num_workers, delay_frame=args.delay_frame)

    # early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)

    model_checkpoint_cb = ModelCheckpoint(dirpath='models/snrep/', filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"), monitor='val_loss', mode='min')

    swa_cb = StochasticWeightAveraging(swa_lrs=1e-2)

    # create trainer
    trainer = pl.Trainer(
        gpus=1 if args.num_gpus == 1 else list(np.arange(int(args.num_gpus))),
        max_epochs=args.max_epochs,
        precision=16,
        callbacks=[model_checkpoint_cb, swa_cb],
        logger=pl_loggers.TensorBoardLogger("lightning_logs/social_nav_rep/"),
        stochastic_weight_avg=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=-1,
        strategy = 'dp'
    )

    trainer.fit(model, dm)
    
    # save model
    torch.save(model.state_dict(), 'models/snrep/' +
               datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')
    
    print('Model has been trained and saved')



