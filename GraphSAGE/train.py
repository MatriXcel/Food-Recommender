import statistics

import dgl

import logging
import torch
import tqdm
import pickle

from data import make_graph
from model import Model
from process import get_recipe_data, get_interactions_data, clean_data, filter_list
from utils import compute_loss_hetero, compute_auc, compute_loss, compute_hinge_loss


def train():
    try:
        hg = dgl.load_graphs('./data/data.bin')
        hg = hg[0][0]
    except:
        logging.info("No graph data file found, creating new graph for training!")
        # Read in data
        recipes = get_recipe_data('data/RAW_recipes.csv')
        interactions = get_interactions_data('data/RAW_interactions.csv')

        logging.info("Read in raw data...Proceeding to processing")

        # Clean data
        interactions, recipes = clean_data(interactions, recipes)

        logging.info("Completed data processing in: {}")

        hg = make_graph(recipes, interactions)

    # fixme for some reason dgl wants all graphs on the cpu? So it'll complain
    hg = hg.to('cpu')

    # training set
    train_eid_dict = {etype: (hg.edges['rated'].data['train_mask'] == True).nonzero(as_tuple=True)[0] for etype in hg.etypes}
    # validation set
    val_eid_dict = {etype: (hg.edges['rated'].data['train_mask'] == False).nonzero(as_tuple=True)[0] for etype in hg.etypes}

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(hg,
                                                train_eid_dict,
                                                sampler,
                                                negative_sampler=dgl.dataloading.negative_sampler.Uniform(2),  # NegativeSampler(hg, 4),
                                                # exclude='reverse_types',
                                                # reverse_eids=torch.cat([torch.arange(hg.num_edges('rated') // 2, hg.num_edges('rated')),
                                                #                         torch.arange(0, hg.num_edges('rated') // 2)]),
                                                batch_size=128,
                                                shuffle=True,
                                                drop_last=False,
                                                device='cpu')

    k = 5
    model = Model(70, 40, 1, hg.etypes)  # .to('cuda:0')
    opt = torch.optim.Adam(model.parameters())
    epoch = 0

    for input_nodes, pos_graph, neg_graph, blocks in tqdm.tqdm(dataloader):
        # if epoch > 0:  # remove epoch to train on entire train dataset
            # move to gpu
            #     blocks = [b.to(torch.device('cuda')) for b in blocks]
            #     pos_graph = pos_graph.to(torch.device('cuda'))
            #     neg_graph = neg_graph.to(torch.device('cuda'))
        model.train()

        node_feat = {'user': blocks[0].srcdata['feat']['user'],
                     'recipe': blocks[0].srcdata['ingredients']['recipe']}

        pos_score, neg_score = model(pos_graph, neg_graph, blocks, node_feat)

        loss = compute_loss_hetero(pos_score[('user', 'rated', 'recipe')],
                                   neg_score[('user', 'rated', 'recipe')])

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            auc = compute_auc(pos_score, neg_score, ('user', 'rated', 'recipe'))
            print("\tLoss: {}\t auc: {}".format(loss.item(), auc))
        epoch += 1

    # save model
    logging.info("Finished Training!")
    print("AUC: {}", auc)

    logging.info("Saving model to disk...")
    with open('data/model.pkl', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logging.info("Exiting..bye!")


if __name__ == '__main__':
    train()
