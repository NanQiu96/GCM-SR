import datetime
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from aggregator import LocalAggregator, GlobalContextExtractor
from torch.nn import Module, Parameter
import torch.nn.functional as F


class CombineGraph(Module):
    def __init__(self, opt, num_node, adj_all, num):
        super(CombineGraph, self).__init__()
        self.opt = opt

        self.batch_size = opt.batch_size
        self.num_node = num_node
        self.dim = opt.hiddenSize
        self.hop = opt.hop
        self.sample_num = opt.n_sample
        self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        self.num = trans_to_cuda(torch.Tensor(num)).float()
        self.scale = opt.scale
        self.norm = opt.norm
        self.tau = opt.tau

        # Aggregator
        self.local_agg = LocalAggregator(self.dim, self.opt.alpha)
        self.global_extractor = GlobalContextExtractor(self.dim, self.hop, self.batch_size, self.sample_num)

        # Item representation & Position representation
        self.embedding = nn.Embedding(num_node, self.dim)
        self.pos_embedding = nn.Embedding(200, self.dim)
        self.type_pos_embedding = nn.Embedding(200, self.dim)

        # Parameters
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu3 = nn.Linear(self.dim, self.dim)
        self.glu4 = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(opt.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def sample(self, target, n_sample):
        # neighbor = self.adj_all[target.view(-1)]
        # index = np.arange(neighbor.shape[1])
        # np.random.shuffle(index)
        # index = index[:n_sample]
        # return self.adj_all[target.view(-1)][:, index], self.num[target.view(-1)][:, index]
        return self.adj_all[target.view(-1)], self.num[target.view(-1)]

    def compute_scores(self, hidden, global_hidden, mask):
        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        type_pos_emb = self.type_pos_embedding.weight[:len]
        type_pos_emb = type_pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        type_hs = torch.sum(global_hidden * mask, -2) / torch.sum(mask, 1)
        type_hs = type_hs.unsqueeze(-2).repeat(1, len, 1)
        type_nh = torch.matmul(torch.cat([type_pos_emb, global_hidden], -1), self.w_3)
        type_nh = torch.tanh(type_nh)
        type_nh = torch.sigmoid(self.glu3(type_nh) + self.glu4(type_hs))
        alpha = torch.matmul(type_nh, self.w_4)
        alpha = alpha * mask
        type_select = torch.sum(alpha * global_hidden, 1)
        
        b = self.embedding.weight[1:]  # n_nodes x latent_size
        
        if self.norm:
            norms = torch.norm(select, p=2, dim=1, keepdim=True)
            select = select.div(norms)
            norms = torch.norm(type_select, p=2, dim=1, keepdim=True)
            type_select = type_select.div(norms)
            norms = torch.norm(b, p=2, dim=1, keepdim=True)
            b = b.div(norms)
        
        scores = torch.matmul(select, b.transpose(1, 0))
        type_scores = torch.matmul(type_select, b.transpose(1, 0))
        if self.scale:
            scores = self.tau * scores  # tau is the sigma factor
            type_scores = self.tau * type_scores  # tau is the sigma factor
        return scores, type_scores

    def forward(self, inputs, adj, mask_item, item):
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)

        # local
        hidden = self.local_agg(h, adj)

        # global context
        item_neighbors = [inputs]
        weight_neighbors = []
        support_size = seqs_len

        for i in range(1, self.hop + 1):
            item_sample_i, weight_sample_i = self.sample(item_neighbors[-1], self.sample_num)
            support_size *= self.sample_num
            item_neighbors.append(item_sample_i.view(batch_size, support_size))
            weight_neighbors.append(weight_sample_i.view(batch_size, support_size))

        neighbor_hiddens = [self.embedding(i) for i in item_neighbors]
        item_hidden = neighbor_hiddens[0]
        neighbor_weight = weight_neighbors

        sess_hiddens = []
        item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1)

        # mean
        sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1)

        sum_item_emb = sum_item_emb.unsqueeze(-2)
        for i in range(self.hop):
            sess_hiddens.append(sum_item_emb.repeat(1, neighbor_hiddens[i].shape[1], 1))

        # extract global context
        global_hidden = self.global_extractor(neighbor_hiddens, neighbor_weight, item_hidden, sess_hiddens)


        return hidden, global_hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data, is_test=False):
    alias_inputs, adj, items, mask, targets, inputs = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()

    hidden, global_hidden = model(items, adj, mask, inputs)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    g_get = lambda index: global_hidden[index][alias_inputs[index]]
    gce_hidden = torch.stack([g_get(i) for i in torch.arange(len(alias_inputs)).long()])
    
    scores, type_scores = model.compute_scores(seq_hidden, gce_hidden, mask)

    if is_test:
        scores = torch.softmax(scores, dim=-1)
        type_scores = torch.softmax(type_scores, dim=-1)

    return targets, scores, type_scores


def train_test(model, train_data, test_data, omega):
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                               shuffle=True, pin_memory=True)
    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        targets, l_scores, type_scores = forward(model, data, is_test=False)
        targets = trans_to_cuda(targets).long()
        l_loss = model.loss_function(l_scores, targets - 1)
        type_loss = model.loss_function(type_scores, targets - 1)
        loss = l_loss + type_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []
    for data in test_loader:
        targets, l_scores, type_scores = forward(model, data, is_test=True)
        scores = (1.0 - omega) * l_scores + omega * type_scores
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit) * 100)
    result.append(np.mean(mrr) * 100)

    return result
