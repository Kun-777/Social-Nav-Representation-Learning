import torch
import torch.nn as nn

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd):
        super(BarlowTwinsLoss, self).__init__()
        self.lambd = lambd
    
    def forward(self, rep_A, rep_B):
        assert rep_A.shape == rep_B.shape
        self.batch_size = rep_A.shape[1]

        # empirical cross-correlation matrix
        c = rep_A.T @ rep_B

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()