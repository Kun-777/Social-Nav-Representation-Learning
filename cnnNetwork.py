import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

class CNNNetwork(nn.Module):
    def __init__(self, joystick_input_size: int, output_size=128):
        # according to the author, Barlow Twins only works well with giant representation vectors
        super(CNNNetwork, self).__init__()
        self.visual_encoder = nn.Sequential(
				nn.Conv2d(5, 32, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(32), nn.ReLU(),  # 200 x 200
				nn.MaxPool2d(kernel_size=2, stride=2),  # 100 x 100
				nn.Conv2d(32, 64, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(64), nn.ReLU(),  # 50 x 50
				nn.MaxPool2d(kernel_size=2, stride=2),  # 25 x 25
				nn.Conv2d(64, 128, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(128), nn.ReLU(),  # 12 x 12
				nn.MaxPool2d(kernel_size=2, stride=2),  # 6 x 6
				nn.Conv2d(128, 256, kernel_size=3, stride=2, bias=False), nn.BatchNorm2d(256), nn.ReLU(),  # 3 x 3
				nn.MaxPool2d(kernel_size=2, stride=2),  # 1 x 1
				nn.Flatten(),
				nn.Linear(256 * 1 * 1, 256), nn.ReLU(),
				nn.Linear(256, 128),
			)

        self.joystick_commands_encoder = nn.Sequential(
            nn.Linear(joystick_input_size, output_size, bias=False), 
            nn.BatchNorm1d(output_size), 
            nn.ReLU(),
            nn.Linear(output_size, output_size, bias=False), 
            nn.BatchNorm1d(output_size), 
            nn.ReLU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, img_batch, joystick_batch, goal_batch):
        batch_size = img_batch.shape[0]

        visual_encodings = self.visual_encoder(img_batch)

        # encode joystick commands
        joy_stick_encodings = self.joystick_commands_encoder(joystick_batch)

        return visual_encodings, joy_stick_encodings
