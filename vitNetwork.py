import numpy as np
import torch
from torch import Tensor
import torch.nn as nn


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


class vitNetwork(nn.Module):
    def __init__(self, joystick_input_size: int, img_size: int, img_channels: int, patch_size: int, embedding_size=128, output_size=128):
        # according to the author, Barlow Twins only works well with giant representation vectors
        super(vitNetwork, self).__init__()
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

        # self.projector = nn.Sequential(
        #     nn.Linear(embedding_size, output_size),
        #     nn.BatchNorm1d(output_size),
        #     nn.ReLU(),
        #     nn.Linear(output_size, output_size),
        #     nn.BatchNorm1d(output_size),
        #     nn.ReLU(),
        #     nn.Linear(output_size, output_size),
        # )

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
        # cls_token = self.projector(cls_token)

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
