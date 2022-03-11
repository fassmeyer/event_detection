import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance_matrix


def permute2st(v, ndim_en=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_en: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(-ndim_en, 0)] + [*range(nd - ndim_en)])


def permute2en(v, ndim_st=1):
    """
    Permute last ndim_en of tensor v to the first
    :type v: torch.Tensor
    :type ndim_st: int
    :rtype: torch.Tensor
    """
    nd = v.ndimension()
    return v.permute([*range(ndim_st, nd)] + [*range(ndim_st)])


def block_diag_irregular(matrices):
    matrices = [permute2st(m, 2) for m in matrices]

    ns = torch.LongTensor([m.shape[0] for m in matrices])
    n = torch.sum(ns)
    batch_shape = matrices[0].shape[2:]

    v = torch.zeros(torch.Size([n, n]) + batch_shape)
    for ii, m1 in enumerate(matrices):
        st = torch.sum(ns[:ii])
        en = torch.sum(ns[:(ii + 1)])
        v[st:en, st:en] = m1
    return permute2en(v, 2)


def compute_adjs_distsim(seq_len, sigma, seq_start_end, traj):
    adj_out = []
    for _, (start, end) in enumerate(seq_start_end):
        # obs_and_pred_traj = torch.cat((obs_traj, pred_traj_gt))
        sim_t = []
        for t in range(0, seq_len):
            dists = distance_matrix(np.asarray(traj[t, start:end, :]),
                                    np.asarray(traj[t, start:end, :]))
            #sum_dist = np.sum(dists)
            #dists = np.divide(dists, sum_dist)
            sim = np.exp(-dists / sigma)
            sim_t.append(torch.from_numpy(sim))
        adj_out.append(torch.stack(sim_t, 0))
    return block_diag_irregular(adj_out)


######################### GOALS UTILITIES ############################
def one_hot_encode(inds, N):
    dims = [inds.size(i) for i in range(len(inds.size()))]
    inds = inds.unsqueeze(-1).long() # (seq_len, num_agents*batch_size, 1)
    dims.append(N) 
    ret = (torch.zeros(dims)).cuda() # -> 0-tensor of shape=(seq_len, num_agents*batch_size, ohe_dim)
    ret.scatter_(-1, inds, 1) # writes the value 1 into ret at the indices specified in the index tensor along the last dimension
    return ret



def to_goals_one_hot(original_goal, ohe_dim):
    return one_hot_encode(original_goal[:, :].data, ohe_dim)


