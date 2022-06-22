from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import Isomap


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0)
        losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)  # 2e-2

        return losses.mean()


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)  # 转置
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def geodesic_distance(x):
    '''
    计算矩阵x和y之间任意两点的测地线距离
    @param x: N×M的矩阵
    @return: 测地线距离矩阵
    '''
    dtype = x.dtype
    x = x.detach().cpu().to(torch.double).numpy()
    isomap = Isomap(n_components=2, n_neighbors=5, path_method="auto")
    isomap.fit(x)
    dist_matrix = isomap.dist_matrix_

    return torch.from_numpy(dist_matrix).to(dtype)


def cdist(x, y):
    """
    Compute distance between each pair of the two collections of inputs.
    :param x: Nxd Tensor
    :param y: Mxd Tensor
    :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
          i.e. dist[i,j] = ||x[i,:]-y[j,:]||
    """
    differences = x.unsqueeze(1) - y.unsqueeze(0)
    distances = torch.sum(differences ** 2, -1).sqrt()
    return distances
