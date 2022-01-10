import torch

import torch.nn.functional as F

from sklearn.metrics import roc_auc_score


# Loss computation
def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


# Heterogeneous graph binary cross entropy loss loss computation
def compute_loss_hetero(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return F.binary_cross_entropy_with_logits(scores.squeeze(1), labels)


# hinge loss
def compute_hinge_loss(pos_score, neg_score):
    # an example hinge loss
    n = pos_score.shape[0]
    return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()


# auc
def compute_auc(pos_score, neg_score, etype):
    # scores = torch.cat([pos_score, neg_score]).numpy()
    scores = torch.cat((pos_score[etype], neg_score[etype])).detach().numpy()
    labels = torch.cat([torch.ones(pos_score[etype].shape[0]),
                        torch.zeros(neg_score[etype].shape[0])]).numpy()
    return roc_auc_score(labels, scores)
