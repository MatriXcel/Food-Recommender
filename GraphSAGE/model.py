
import dgl
import torch
import torch.nn.functional as F

import dgl.function as nn

import numpy as np
import pandas as pd

from dgl.nn.pytorch import HeteroGraphConv, GraphConv


class RGCN(torch.nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = HeteroGraphConv({rel: GraphConv(in_feats, hid_feats, norm='right') for rel in rel_names},
                                     aggregate='sum')
        self.conv2 = HeteroGraphConv({rel: GraphConv(hid_feats, out_feats, norm='right') for rel in rel_names},
                                     aggregate='sum')

    def forward(self, blocks, x):
        x = self.conv1(blocks[0], x)
        x = {k: F.leaky_relu(v) for k, v in x.items()}
        x = self.conv2(blocks[1], x)

        return x


class Model(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, etypes):
        super().__init__()

        self.rgcn = RGCN(in_feats, hidden_feats, out_feats, etypes)
        self.pred = ScorePredictor()

    def forward(self, pos_g, neg_g, blocks, x):
        x = self.rgcn(blocks, x)
        pos_score, neg_score = self.pred(pos_g, x), self.pred(neg_g, x)

        return pos_score, neg_score


class ScorePredictor(torch.nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            # 'x' contains the node representations computed previously
            edge_subgraph.ndata['x'] = x

            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class HeteroDotProductPredictor(torch.nn.Module):
    ''' Dot predictor for heterograph edges '''
    def forward(self, graph, h, etype):
        with graph.local_scope():
            # 'h' contains the node representations computed previously
            graph.ndata['h'] = h
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'), etype=etype)

            return graph.edges[etype].data['score']
