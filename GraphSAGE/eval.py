import argparse
import dgl
import logging
import pickle
import statistics
import tqdm

from utils import compute_auc


def eval():
    with open('data/model.pkl', 'rb') as pfile:
        model = pickle.load(pfile)
    hg = dgl.load_graphs('data/data.bin')[0][0]

    # validation set
    val_eid_dict = {etype: (hg.edges['rated'].data['train_mask'] == False).nonzero(as_tuple=True)[0] for etype in hg.etypes}

    logging.info("Evaluating...")

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    val_dataloader = dgl.dataloading.EdgeDataLoader(hg,
                                                    val_eid_dict,
                                                    sampler,
                                                    negative_sampler=dgl.dataloading.negative_sampler.Uniform(2),
                                                    batch_size=128,
                                                    shuffle=True,
                                                    drop_last=False,
                                                    device='cpu')

    total_auc = []
    for input_nodes, pos_graph, neg_graph, blocks in tqdm.tqdm(val_dataloader):
        model.eval()
        node_feat = {'user': blocks[0].srcdata['feat']['user'],
                     'recipe': blocks[0].srcdata['ingredients']['recipe']}

        pos_score, neg_score = model(pos_graph, neg_graph, blocks, node_feat)

        total_auc.append(compute_auc(pos_score, neg_score, ('user', 'rated', 'recipe')))

    print("AUC: {}".format(statistics.mean(total_auc)))



if __name__ == '__main__':
    eval()