import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy


class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


class GlobalContextExtractor(nn.Module):
    def __init__(self, dim, hop, batch_size, sample_num, name=None):
        super(GlobalContextExtractor, self).__init__()
        self.dim = dim
        self.hop = hop
        self.batch_size = batch_size
        self.sample_num = sample_num

        self.global_agg = []
        for i in range(self.hop):
            agg = GlobalAggregator(self.dim)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)

        self.w_1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, self.dim))

    def forward(self, neighbor_hiddens, neighbor_weights, item_hidden, sess_hiddens):
        # aggregate
        for n_hop in range(self.hop):
            neighbor_next_hiddens = []
            for hop in range(self.hop - n_hop):
                aggregator = self.global_agg[n_hop]
                neighbor_hidden = neighbor_hiddens[hop+1]
                neighbor_hidden = neighbor_hidden.view(neighbor_hidden.shape[0], -1, self.sample_num, self.dim)
                neighbor_weight = neighbor_weights[hop]
                neighbor_weight = neighbor_weight.view(neighbor_weight.shape[0], -1, self.sample_num)
                next_hidden = aggregator(neighbor_hidden=neighbor_hidden,
                                    neighbor_weight=neighbor_weight,
                                    sess_hidden=sess_hiddens[hop])
                neighbor_next_hiddens.append(next_hidden)
            neighbor_hiddens = neighbor_next_hiddens

        agg_hidden = neighbor_hiddens[0].view(neighbor_hiddens[0].shape[0], -1, self.dim)

        # gated network
        g_weight = torch.sigmoid(torch.matmul(item_hidden, self.w_1) + torch.matmul(agg_hidden, self.w_2))
        output = (1.0 - g_weight) * item_hidden + g_weight * agg_hidden

        return output


class GlobalAggregator(nn.Module):
    def __init__(self, dim, name=None):
        super(GlobalAggregator, self).__init__()
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))

    def forward(self, neighbor_hidden, neighbor_weight, sess_hidden):
        alpha = torch.matmul(torch.cat([sess_hidden.unsqueeze(2).repeat(1, 1, neighbor_hidden.shape[2], 1) * 
                    neighbor_hidden, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
        alpha = torch.softmax(alpha, -1).unsqueeze(-1)
        agg_hidden = torch.sum(alpha * neighbor_hidden, dim=-2)

        return agg_hidden
