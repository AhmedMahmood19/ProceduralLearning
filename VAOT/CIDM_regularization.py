import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_distance_matrix(x, y):
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)
    dist = torch.pow(x - y, 2).sum(3)
    return dist

class Contrastive_IDM(nn.Module):
    def __init__(self, sigma, margin):
        super(Contrastive_IDM, self).__init__()
        self.sigma = sigma
        self.margin = margin

    def forward(self, a_emb, b_emb, a_idx, b_idx, a_len, b_len):
        
        dist_a = calc_distance_matrix(a_emb, a_emb).squeeze(0)
        dist_b = calc_distance_matrix(b_emb, b_emb).squeeze(0)

        idm_a = self._compute_idm(dist_a, a_idx, a_len)
        idm_b = self._compute_idm(dist_b, b_idx, b_len)

        return idm_a + idm_b

    def _compute_idm(self, dist, idx, seq_len):
        grid_x, grid_y = torch.meshgrid(idx, idx, indexing='ij')

        prob = F.relu(self.margin - dist)

        weights_orig = 1 + torch.pow(grid_x - grid_y, 2)
        diff = torch.abs(grid_x - grid_y) - (self.sigma / seq_len)

        weights_neg = torch.where(diff > 0, weights_orig, torch.zeros_like(diff))
        weights_pos = torch.where(diff > 0, torch.zeros_like(diff), torch.ones_like(diff))

        idm = weights_neg * prob + weights_pos * dist

        return torch.sum(idm)
