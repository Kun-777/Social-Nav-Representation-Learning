"""
Author: Abhinav Chadaga
"""
import torch
import torch.nn as nn
from torch import Tensor

class PatchEmbedding(nn.Module):
    """ Convert a 2D image into 1D patches and embed them
    """

    def __init__(self, img_size: int, input_channels: int, patch_size=16, embedding_size=1280) -> \
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

class BTSNNetwork(nn.Module):
    def __init__(self, joystick_input_size: int, img_size: int, img_channels: int, patch_size=16, embedding_size=1280, output_size=100):
        super(BTSNNetwork, self).__init__()
        self.patch_embed = PatchEmbedding(img_size=img_size, input_channels=img_channels, patch_size=patch_size, embedding_size=embedding_size)

        # class token from BERT
        # contains all learned information
        self.cls_token = nn.Parameter(torch.zeros((1, 1, embedding_size)))

        # learnable positional embeddings for each patch
        self.positional_embeddings = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embedding_size))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=8, activation='gelu', batch_first=True)
        self.layer_norm = nn.LayerNorm(embedding_size, eps=1e-6)
        self.visual_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=6, norm=self.layer_norm)
        # MLP head, only uses the cls token
        self.mlp_head = nn.Linear(embedding_size, output_size)

        self.joystick_commands_encoder = nn.Sequential(
            nn.Linear(joystick_input_size, joystick_input_size), nn.BatchNorm1d(joystick_input_size), nn.ReLU(),
            nn.Linear(joystick_input_size, output_size), nn.BatchNorm1d(output_size), nn.ReLU(),
        )

    def forward(self, img_batch, joystick_batch):
        batch_size = img_batch.shape[0]
        # turn batch of images into embeddings
        img_batch = self.patch_embed(img_batch)
        # expand cls token from 1 batch
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        # concatenate cls token to beginning of patch embeddings
        img_batch = torch.cat((cls_token, img_batch), dim=1)
        # add learnable positional embeddings
        img_batch += self.positional_embeddings
        # pass input with cls token and positional embeddings through the transformer encoder
        visual_encoding = self.visual_encoder(img_batch)
        # keep only cls token, discard rest
        cls_token = visual_encoding[:, 0]
        # pass cls token into MLP head
        cls_token = self.mlp_head(cls_token)

        # encode joystick commands
        joy_stick_encodings = self.joystick_commands_encoder(joystick_batch)

        return cls_token, joy_stick_encodings