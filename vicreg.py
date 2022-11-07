import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch import optim, nn
import numpy as np

class VICReg(nn.Module):
    def __init__(self, num_features=128, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y):
        self.batch_size = x.shape[0]
        repr_loss = F.mse_loss(x, y)
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(x), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.001)
        std_y = torch.sqrt(y.var(dim=0) + 0.001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

class W_VICReg(nn.Module):
    def __init__(self, num_features=128, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, weights):
        self.batch_size = x.shape[0]
        repr_loss = F.mse_loss(x, y)
        # repr_loss = ((((x-y)**2).transpose(0,1) * w).transpose(0,1)).mean()
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(x), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(weighted_variance(x, weights) + 0.001)
        std_y = torch.sqrt(weighted_variance(y, weights) + 0.001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.batch_size - 1)
        cov_y = (y.T @ y) / (self.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss

class M_VICReg(nn.Module):
    def __init__(self, num_features=128, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0):
        super().__init__()
        self.num_features = num_features
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def forward(self, x, y, masks):
        self.batch_size = x.shape[0]
        # masks = np.asarray([1 for _ in range(self.batch_size)])
        # straight = np.asarray([1.6, 0, 0] * 150)
        # for i, joy in enumerate(fixed_joy):
        #     dist = ((joy - straight)**2).mean()
        #     if dist < 0.005:
        #         masks[i] = 0
        #         break
        # x = x[masks == 1]
        # y = y[masks == 1]
        repr_loss = F.mse_loss(x, y)
        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(x), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.001)
        std_y = torch.sqrt(y.var(dim=0) + 0.001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (x.shape[0] - 1)
        cov_y = (y.T @ y) / (y.shape[0] - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        
        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def weighted_variance(x, weights):
    # x.shape should be (batch_size, d), weights.shape should be (batch_size, 1)
    ret = torch.clone(x)
    mean = x.mean(dim=0)
    v_1 = weights.sum()
    v_2 = weights.pow(2).sum()
    ret = (v_1 / (v_1 ** 2 - v_2)) * (((ret - mean).pow_(2).transpose(0,1) * weights).transpose(0,1)).sum(dim=0)
    return ret
