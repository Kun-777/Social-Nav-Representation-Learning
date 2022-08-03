from unittest.mock import patch
import cv2
import numpy as np
import torch
import argparse
import os
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from datetime import datetime
import glob
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pickle
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import matplotlib.pyplot as plt
# from scripts.utils import get_mask
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

class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size: int, embedding_size: int) -> \
            None:
        """ Initialize a PatchEmbedding Layer
        :param img_size: size of the input images (assume square)
        :param input_channels: number of input channels (1 for grayscale, 3 for RGB)
        :param patch_size: size of a 2D patch (assume square)
        :param embedding_size: size of the embedding for a patch (input to the transformer)
        """
        super(PatchEmbedding, self).__init__()
        self.input_channels = input_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # linearly transform patches
        self.lin_proj = nn.Linear(in_features=(self.patch_size ** 2) * self.input_channels, out_features=embedding_size)

    def forward(self, x: Tensor) -> Tensor:
        """ Split a batch of images into patches and linearly embed each patch
        :param x: input tensor (batch_size, channels, img_height, img_width)
        :return: a batch of patch embeddings (batch_size, num_patches, embedding_size)
        """
        # sanity checks
        assert len(x.shape) == 4
        batch_size, num_channels, height, width = x.shape
        assert height == width and height == self.img_size

        # batch_size, channels, v slices, h slices, patch_size ** 2
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # combine vertical and horizontal slices
        x = x.reshape(x.shape[0], x.shape[1], -1, self.patch_size, self.patch_size)
        x = x.movedim(1, -1)  # batch_size, num patches p channel, patch_size ** 2, channels
        x = x.flatten(-3)  # 3D patch to 1D patch vector
        x = self.lin_proj.forward(x)  # linear transformation
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class SNNetwork(nn.Module):
    def __init__(self, joystick_input_size: int, img_size: int, img_channels: int, patch_size: int, embedding_size=128, output_size=128):
        # according to the author, Barlow Twins only works well with giant representation vectors
        super(SNNetwork, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, input_channels=img_channels, patch_size=patch_size, embedding_size=embedding_size)

        # class token from BERT
        # contains all learned information
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embedding_size)))

        # learnable positional embeddings for each patch
        self.positional_embeddings = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embedding_size))

        # goal position encoder
        self.goal_embed = nn.Linear(2, embedding_size)

        # transformer encoder
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.visual_encoder = nn.ModuleList([Block(dim=embedding_size, num_heads=8, norm_layer=nn.LayerNorm) for i in range(6)])

        # MLP head and normalization for only the cls token
        self.mlp_head = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.ReLU(), nn.Linear(embedding_size, output_size))

        self.joystick_commands_encoder = nn.Sequential(
            nn.Linear(joystick_input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
            nn.Linear(output_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, img_batch, joystick_batch, goal_batch):
        batch_size = img_batch.shape[0]
        # turn batch of images into embeddings
        img_batch = self.patch_embed(img_batch)
        # expand cls token from 1 batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # concatenate cls token to beginning of patch embeddings
        img_batch = torch.cat((cls_token, img_batch), dim=1)
        # add learnable positional embeddings
        img_batch += self.positional_embeddings
        # concatenate goal embedding to end of patch embeddings
        img_batch = torch.cat((img_batch, self.goal_embed(goal_batch).unsqueeze(1)), dim=1)
        # pass input with cls token and positional embeddings through the transformer encoder
        for block in self.visual_encoder:
            img_batch = block(img_batch)
        visual_encoding = self.layer_norm(img_batch)
        # keep only cls token, discard rest
        cls_token = visual_encoding[:, 0]
        # pass cls token into MLP head
        cls_token = self.mlp_head(cls_token)

        # encode joystick commands
        joy_stick_encodings = self.joystick_commands_encoder(joystick_batch)

        return cls_token, joy_stick_encodings

    def get_last_selfattention(self, img_batch, goal_batch):
        batch_size = img_batch.shape[0]
        # turn batch of images into embeddings
        img_batch = self.patch_embed(img_batch)
        # expand cls token from 1 batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # concatenate cls token to beginning of patch embeddings
        img_batch = torch.cat((cls_token, img_batch), dim=1)
        # add learnable positional embeddings
        img_batch += self.positional_embeddings
        # concatenate goal embedding to end of patch embeddings
        img_batch = torch.cat((img_batch, self.goal_embed(goal_batch).unsqueeze(1)), dim=1)

        # we return the output tokens from the `n` last blocks
        for i, block in enumerate(self.visual_encoder):
            if i < len(self.visual_encoder) - 1:
                img_batch = block(img_batch)
            else:
                return block(img_batch, return_attention=True)


class SNModel(pl.LightningModule):
    def __init__(self, lambd, lr, weight_decay, patch_size):
        super(SNModel, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.patch_size = patch_size

        # self.batch_size = batch_size
        # self.data_path = full_args.data_path

        # self.save_hyperparameters(
        #     'full_args'
        # )

        self.model = SNNetwork(joystick_input_size=900, img_size=400, img_channels=5, patch_size=patch_size)
        self.model.to(self.device)
        self.barlow_twins_loss = BarlowTwinsLoss(lambd)
        self.vicreg_loss = VICReg()

    def forward(self, img_batch, joystick_batch, goal_patch):
        return self.model(img_batch, joystick_batch, goal_patch)
    
    def training_step(self, batch, batch_idx):
        img_batch, joystick_batch, goal_batch = batch
        img_rep, joystick_rep = self.forward(img_batch.float(), joystick_batch.float(), goal_batch.float())
        # loss = self.barlow_twins_loss(img_rep, joystick_rep)
        loss = self.vicreg_loss(img_rep, joystick_rep)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img_batch, joystick_batch, goal_batch = batch
        img_rep, joystick_rep = self.forward(img_batch.float(), joystick_batch.float(), goal_batch.float())
        # loss = self.barlow_twins_loss(img_rep, joystick_rep)
        loss = self.vicreg_loss(img_rep, joystick_rep)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        img_batch, _, goal_batch = batch
        attentions = self.model.get_last_selfattention(img_batch=img_batch.float(), goal_batch=goal_batch.float())
        nh = attentions.shape[1] # number of head
        # we keep only the output patch attention
        numfeat = img_batch.shape[-1] // self.patch_size
        attentions = attentions[0, :, 1:-1, 1:-1].reshape(nh, numfeat**2, numfeat**2)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=self.patch_size, mode="nearest")[0].cpu().detach().numpy()
        plt.imshow(attentions[0], cmap=plt.cm.gray)
        

    
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

        # flattened_traj = []
        # for trajectory_val in self.data['human_expert_odom'][idx+20-1][:100]:
        #     # temp = []
        #     for single_traj in trajectory_val:
        #         if type(single_traj) is list:
        #             for orientation in single_traj:
        #                 # temp.append(orientation)
        #                 flattened_traj.append(orientation)
        #         else:
        #             # temp.append(single_traj)
        #             flattened_traj.append(single_traj)
        #     # flattened_traj.append(temp)

        # future_trajectory = np.asarray(flattened_traj)

        # relative_goal_pos = np.asarray([self.data['human_expert_odom'][idx+20-1][-1][0]-self.data['human_expert_odom'][idx+20-1][0][0],
        #                                 self.data['human_expert_odom'][idx+20-1][-1][1]-self.data['human_expert_odom'][idx+20-1][0][1]])
        relative_goal_pos = np.asarray([self.data['local_goal_human_odom'][idx+20-1][-1][0],self.data['local_goal_human_odom'][idx+20-1][-1][1]])

        return bev_img_stack, future_joystick_values, relative_goal_pos

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learn representations from SCAND dataset using Barlow Twins\' model.')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
    parser.add_argument('--num_gpus', type=int, default=4, help='Number of GPUS to use')
    parser.add_argument('--delay_frame', type=int, default=0, help='Number of initial frames to skip')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers to use')
    parser.add_argument('--patch_size', type=int, default=16, help='patch size')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Multiplier for batch size')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lambd', type=float, default=0.0051, metavar='L', help='weight on off-diagonal terms for Barlow Twins loss')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--notes', type=str, default='', help='notes for this specific run')
    parser.add_argument('--checkpoint', type=str, default='None')
    parser.add_argument('--use_pretrained_weight', type=str, default="")
    args = parser.parse_args()

    # check if data path exists
    if not os.path.exists(args.data_path):
        raise Exception('Data path does not exist')

    # check if checkpoint path exists
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    # create model
    model = SNModel(lambd=args.lambd, lr=args.lr, weight_decay=args.weight_decay, patch_size=8)

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

    early_stopping_cb = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.00, patience=100)

    model_checkpoint_cb = ModelCheckpoint(dirpath='models/snrep/', filename=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"), monitor='val_loss', mode='min')

    # create trainer
    trainer = pl.Trainer(
        gpus=1 if args.num_gpus == 1 else list(np.arange(int(args.num_gpus))),
        max_epochs=args.max_epochs,
        callbacks=[early_stopping_cb, model_checkpoint_cb],
        # logger=pl_loggers.TensorBoardLogger("lightning_logs/global_local_planner/"),
        # distributed_backend='gloo',
        stochastic_weight_avg=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=-1,
        # replace_sampler_ddp=False,
        progress_bar_refresh_rate=1
    )

    trainer.fit(model, dm)
    
    # save model
    torch.save(model.state_dict(), 'models/snrep/' +
               datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + '.pt')
    
    print('Model has been trained and saved')



