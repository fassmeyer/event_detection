import sys
import argparse
import pathlib
import torch
import random
import logging
import numpy as np
from math import ceil
from time import time
from datetime import timedelta
import matplotlib.pyplot as plt
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from model import DetNet
from loader_soccer import data_loader
from loader_soccer import seq_collate
from training_utils import to_goals_one_hot
from adjacency_matrix import compute_adjs_distsim, compute_adjs_knnsim, compute_adjs



parser = argparse.ArgumentParser()

# Dataset options
parser.add_argument('--num_workers', default=1, type=int, required=False, help='Number of workers for loading data')
parser.add_argument('--obs_len', default=10, type=int, required=False, help='Timesteps of observation')
parser.add_argument('--pred_len', default=40, type=int, required=False, help='Timesteps of prediction')
parser.add_argument('--players', type=str, choices=['atk', 'def', 'all'], required=True, help='Which players to use')
parser.add_argument('--unsupervised_percentage', type=float, default=1, help='Percentage of unsupervised data from training data')


# Optimization options
parser.add_argument('--learning_rate', default=1e-3, type=float, required=False, help='Initial learning rate')
parser.add_argument('--batch_size', default=32, type=int, required=False, help='Batch size')
parser.add_argument('--num_epochs', default=100, type=int, required=False, help='Training epochs number')

# Model
parser.add_argument('--n_layers', default=2, type=int, required=False, help='Number of recurrent layers')
parser.add_argument('--x_dim', default=2, type=int, required=False, help='Dimension of the input of the single agent')
parser.add_argument('--h_dim', default=64, type=int, required=False, help='Dimension of the hidden layers')
parser.add_argument('--z_dim', default=32, type=int, required=False, help='Dimension of the latent variables')
parser.add_argument('--g_dim', default=4, type=int, required=False, help='Dimension of the goal variables')
parser.add_argument('--rnn_dim', default=64, type=int, required=False, help='Dimension of the recurrent layers')


# Miscellaneous
parser.add_argument('--seed', default=128, type=int, required=False, help='PyTorch random seed')
parser.add_argument('--run', required=True, type=str, help='Current run name')

# graph
parser.add_argument('--graph_model', type=str, required=True, choices=['gat','gcn'], help='Graph type')
parser.add_argument('--graph_hid', type=int, default=8, help='Number of hidden units')
parser.add_argument('--sigma', type=float, default=1.2, help='Sigma value for similarity matrix')
parser.add_argument('--adjacency_type', type=int, default=2, choices=[0,1,2], help='Type of adjacency matrix: '
                                                                                   '0 (fully connected graph),'
                                                                                   '1 (distances similarity matrix),'
                                                                                   '2 (knn similarity matrix).')
parser.add_argument('--top_k_neigh', type=int, default=None)


# GAT-specific
parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for graph attention network')
parser.add_argument('--alpha', type=float, default=0.2, help='Negative steep for the Leaky-ReLU activation')



# dirs
ROOT_DIR = pathlib.Path('.').absolute().parent
BASE_DIR = ROOT_DIR / 'runs' 
DATASET_DIR = BASE_DIR.parent.absolute() / 'datasets' 


# global vars
FORMAT = '[%(levelname)s | %(asctime)s]: %(message)s'
DATEFMT = '%Y/%m/%d %H:%M'
DEBUG = False



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args, epch, train_loader, model, optimizer, n_agents):
    train_loss = 0

    start = time()
    n_processed_batches = 0
    
    model.train()
    for batch_idx, data in enumerate(train_loader):
        # 1) get mini-batches for labels + trajectory data
        data = [tensor.cuda() for tensor in data]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt,
         obs_goals, pred_goals_gt, seq_start_end) = data
        
        ###################### assign negative labels to the first 20 timesteps ##############################
        obs_goals[:,:] = 0
        pred_goals_gt[:10, :] = 0
        ###########################################################################################

        seq_len = len(obs_traj) + len(pred_traj_gt)
        # goals one-hot encoding
        obs_goals_ohe = to_goals_one_hot(obs_goals, args.g_dim).cuda()
        pred_goals_gt_ohe = to_goals_one_hot(pred_goals_gt, args.g_dim).cuda()

        # during training, we feed the entire trjs to the model
        all_traj = torch.cat((obs_traj, pred_traj_gt), dim=0)
        all_traj_rel = torch.cat((obs_traj_rel, pred_traj_rel_gt), dim=0)
        all_goals_ohe = torch.cat((obs_goals_ohe, pred_goals_gt_ohe), dim=0)
        
        
        # 2) Compute adjacency matrix for current mini-batch 
        if args.adjacency_type == 0:
            adj_out = compute_adjs(args, seq_start_end).cuda()
        elif args.adjacency_type == 1:
            adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj.detach().cpu(), pred_traj_gt.detach().cpu()).cuda()
        elif args.adjacency_type == 2:
            adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj.detach().cpu(), pred_traj_gt.detach().cpu()).cuda()

        # if batch_idx % int(1/args.unsupervised_percentage) != 0:
        #     continue
    
        n_processed_batches += 1
        
        # 3) forward + backward step 
        optimizer.zero_grad()
        loss, _ = model(all_traj_rel, all_goals_ohe, adj_out,n_agents)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        

    end = time()
    elapsed = str(timedelta(seconds=(ceil(end - start))))
    logging.info('Epoch [{}], time elapsed: {}'.format(epch, elapsed))




def validate(args, epch, valid_loader, model, n_agents):
    test_loss = 0
    n_batches = 0
    pred_probs = []
    acc = []
    current_best = 0
    ground_truth = []

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(valid_loader):
            # 1) agent labels + multi-agent trajectories
            n_batches += 1
            data = [tensor.cuda() for tensor in data]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_rel_gt,
             obs_goals, pred_goals_gt, seq_start_end) = data
            
            ###################### assign negative labels to the first 20 timesteps ##############################
            obs_goals[:,:] = 0
            pred_goals_gt[:10, :] = 0
            ###########################################################################################

            
            # print("Validation:", pred_traj_rel_gt.shape, pred_goals_gt.shape)
            # goals one-hot encoding
            obs_goals_ohe = to_goals_one_hot(obs_goals, args.g_dim).cuda() # (batch_size, num_classes)
            pred_goals_ohe = to_goals_one_hot(pred_goals_gt, args.g_dim).cuda()


            # 2) adj matrix for current batch
            if args.adjacency_type == 0:
                adj_out = compute_adjs(args, seq_start_end).cuda()
            elif args.adjacency_type == 1:
                adj_out = compute_adjs_distsim(args, seq_start_end, obs_traj.detach().cpu(),
                                               pred_traj_gt.detach().cpu()).cuda()
            elif args.adjacency_type == 2:
                adj_out = compute_adjs_knnsim(args, seq_start_end, obs_traj.detach().cpu(),
                                              pred_traj_gt.detach().cpu()).cuda()
            
            # 3) forward propagation + loss computation
            loss, h = model(obs_traj_rel, obs_goals_ohe, adj_out, n_agents)
            test_loss += loss.item()

            # compute accuracy for future steps from latest h, latest gt goal
            predicted_probs = model.prediction(h, pred_traj_rel_gt, adj_out, n_agents) # (40, batch_size, num_classes)
            pred_probs += [predicted_probs.detach().cpu().numpy()]
            ground_truth += [pred_goals_gt.detach().cpu().numpy()]

            pred = torch.max(predicted_probs, dim=-1, keepdim=True)[1].cuda() # (40, batch_size): maximum over class dimension for each timestep and sequence
            mini_batch_acc = torch.sum(torch.eq(pred, pred_goals_gt.unsqueeze(-1)))/(pred.shape[0]*pred.shape[1])
            acc += [mini_batch_acc]


            ############################################################################################################
            if (mini_batch_acc.item() > current_best) and (torch.sum(pred_goals_gt) > 0):
                # print(mini_batch_acc.item())
                # fn = 'pred_probabilities.npy'
                arr = predicted_probs.cpu().detach().numpy()
                # np.save(fn, arr)
                # logging.info('Saved best predicted probs to' + fn)
                current_best = mini_batch_acc.item()
            ##########################################################################################################
        
        acc_val = torch.sum(torch.tensor(acc))/n_batches # average over mini-batch accuracies
        val_predictions = np.concatenate(pred_probs, axis=1)
        gt = np.concatenate(ground_truth, axis=1)

    return acc_val, arr, val_predictions, gt


def main(args):
    # current run main directory
    curr_run_dir = BASE_DIR / '{}'.format(args.run)
    curr_run_dir.mkdir(parents=True, exist_ok=True)

    # plain text logs
    log_full_path = curr_run_dir.absolute() / '{}.log'.format(args.run)
    logging.basicConfig(filename=log_full_path, level=logging.INFO,format=FORMAT, datefmt=DATEFMT)


    # load goals + trajectory data
    logging.info('Loading training/test sets...')
    train_set, train_loader = data_loader(args, DATASET_DIR, 'train')
    valid_set, valid_loader = data_loader(args, DATASET_DIR, 'validation')
    # test_set, test_loader = data_loader(args, DATASET_DIR, 'test')
    n_max_agents = max(train_set.__max_agents__(), valid_set.__max_agents__())

    # consider only a subset for training dependent on the specified %
    # example: 1% of training data -> Consider every 1/0.01=100th sequence
    # print("Total number of sequences:", len(train_set))
    mask = list(range(0, len(train_set), int(1/args.unsupervised_percentage)))
    # print("Considered sequences:", len(mask))
    train_set = torch.utils.data.Subset(train_set, mask)
    train_loader =  torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=seq_collate)
    
    

    # saves directory
    save_dir = curr_run_dir / 'saves'
    save_best_dir = save_dir / 'best'
    eval_save_dir = save_dir / 'eval'
    save_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None
    save_best_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None
    eval_save_dir.mkdir(parents=True, exist_ok=True) if not DEBUG else None


    # model, optim, lr scheduler
    model = DetNet(args).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        threshold=1e-2,
        patience=10,
        factor=5e-1,
        verbose=True
    )

    best_acc = 0
    for epoch in range(1, args.num_epochs+1):
        train(args, epoch, train_loader, model, optimizer, n_max_agents)
        acc, arr_probs, val_predictions, val_gt = validate(args, epoch, valid_loader, model, n_max_agents)
        logging.info('Validation ACC: {} [epoch: {}]'.format(acc.item(), epoch))

        if 0 <= acc.item() >= best_acc:
            fn = '{}/checkpoint_epoch_{}.pth'.format(save_best_dir, str(epoch))
            best_acc = acc.item()
            for child in save_best_dir.glob('*'):
                if child.is_file():
                    child.unlink()  # delete previous best
            torch.save(model.state_dict(), fn)
            logging.info('Saved best checkpoint to ' + fn)

            # # save mini-batch probabilities for plotting
            # fn_probs = 'pred_probabilities.npy'
            # np.save(fn_probs, arr_probs)
            # logging.info('Saved best predicted probs to' + fn_probs)

            # # save validation probabilities and ground truths for computing F1 score
            # fn_val_probs = 'validation_probabilities.npy'
            # np.save(fn_val_probs, val_predictions)
            # fn_gt = 'validation_ground_truth.npy'
            # np.save(fn_gt, val_gt)





if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seed(args.seed)
    DEBUG = (sys.gettrace() is not None)
    main(args)