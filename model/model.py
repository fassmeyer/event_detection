import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from graph_network import GAT



class DetNet(nn.Module):
    def __init__(self, args):
        super(DetNet, self).__init__()

        self.n_layers = args.n_layers
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.rnn_dim = args.rnn_dim

        # GNN hyperparameters
        self.graph_model = args.graph_model
        self.graph_hid = args.graph_hid
        self.adjacency_type = args.adjacency_type
        self.top_k_neigh = args.top_k_neigh
        self.sigma = args.sigma
        self.alpha = args.alpha
        self.n_heads = args.n_heads
        # self.head_type = head_type

        self.g_dim = args.g_dim # num of classes

        # hiddens graph
        if self.adjacency_type == 2 and self.top_k_neigh is None:
            raise Exception(
                'Using KNN-similarity but top_k_neigh is not specified')
        if self.graph_model == 'gcn':
            raise(Exception('Model does not support GCN networks.'))
            # takes as input a set of hidden states and updates them based on neighboring information
            # self.graph_hiddens = GCN(
            #     self.rnn_dim, self.graph_hid, self.rnn_dim)
        elif self.graph_model == 'gat':
            assert self.n_heads is not None
            assert self.alpha is not None
            self.graph_hiddens = GAT(
                self.rnn_dim, self.graph_hid, self.rnn_dim, self.alpha, self.n_heads)

        # mapping original hidden states and refined hidden states to new hidden statess
        self.lg_hiddens = nn.Linear(self.rnn_dim + self.rnn_dim, self.rnn_dim)

        # feature extractor
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, self.h_dim),
            nn.LeakyReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU()
        )

        # recurrence
        self.rnn = nn.GRU(self.h_dim, self.rnn_dim, self.n_layers)

        if self.g_dim > 1:
            classify_activation = nn.Softmax(dim=-1)
            self.loss = nn.CrossEntropyLoss()
        else:
            classify_activation = nn.Sigmoid()
            self.loss = nn.BCELoss()

        self.classify = nn.Sequential(
            nn.Linear(self.rnn_dim, self.g_dim),
            classify_activation
        )

    

    def forward(self, traj_rel, targets, adj, n_agents):
        seq_len, batch, features = traj_rel.shape

        batch_size = int(batch/n_agents)
        ball_indices = [idx*n_agents for idx in range(batch_size)]

        h = torch.zeros(self.n_layers, batch, self.rnn_dim).cuda()
        
        loss = torch.zeros(1).cuda()

        for t in range(1, seq_len):
            g_t = targets[t] # (n_sequences, num_classes)

            # step 1: agent recurrence -> shape = (num_agents*batch_size, 2) -> (num_agents*batch_size, rnn_dim)
            x_t = traj_rel[t]
            phi_x_t = self.phi_x(x_t) 
            _, h = self.rnn(phi_x_t.unsqueeze(0), h)
            
            # step 2: hidden states refinement with GNN -> (num_agents*batch_size, rnn_dim)
            h_refined = self.graph_hiddens(h[-1].clone(), adj[t])

            # step 3: final hidden state via linear projection -> (num_agents*batch_size, rnn_dim)
            h[-1] = self.lg_hiddens(torch.cat((h_refined,
                                    h[-1]), dim=-1)).unsqueeze(0)
            
            # step 4: estimate categorical distribution over label space -> (num_agents*batch_size, num_classes)
            probs = self.classify(h[-1])
            # we are only interested in the ball predictions
            ball_probs = probs[ball_indices] # (n_sequences, num_classes)
            p_g_t = D.OneHotCategorical(ball_probs)
            
            loss -= torch.mean(p_g_t.log_prob(g_t))

        return loss, h
    
    
    
    def prediction(self, h, traj_rel, adj, n_agents):
        seq_len, batch, features = traj_rel.shape

        batch_size = int(batch/n_agents)
        ball_indices = [idx*n_agents for idx in range(batch_size)]
        pred_probs = torch.zeros(seq_len, batch_size, self.g_dim).cuda()
        
        with torch.no_grad():
            for t in range(seq_len):
                
                # step 1: agent recurrence -> shape = (num_agents*batch_size, 2) -> (num_agents*batch_size, rnn_dim)
                x_t = traj_rel[t]
                phi_x_t = self.phi_x(x_t) 
                _, h = self.rnn(phi_x_t.unsqueeze(0), h)
                
                # step 2: hidden states refinement with GNN -> (num_agents*batch_size, rnn_dim)
                h_refined = self.graph_hiddens(h[-1].clone(), adj[t])

                # step 3: final hidden state via linear projection -> (num_agents*batch_size, rnn_dim)
                h[-1] = self.lg_hiddens(torch.cat((h_refined,
                                    h[-1]), dim=-1)).unsqueeze(0)
            
                p_g_t = self.classify(h[-1])
                p_g_t_ball = p_g_t[ball_indices]
                pred_probs[t] += p_g_t_ball
        
        return pred_probs
            