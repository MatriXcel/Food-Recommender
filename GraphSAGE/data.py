import dgl
import torch
import pandas as pd

from dgl.data.utils import save_graphs

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess


def make_graph(recipes: pd.DataFrame, interactions: pd.DataFrame):
    """
        Try to build a (heterogeneous) graph, use the
        columns with sorted contiguous IDs for recipes and user IDs
    """
    hg = dgl.heterograph({
        ('user', 'rated', 'recipe'): (torch.LongTensor(interactions.user_id_contig.values),
                                      torch.LongTensor(interactions.recipe_id_contig.values)),
        ('recipe', 'isRated', 'user'): (torch.flip(torch.LongTensor(interactions.recipe_id_contig.values), dims=[0]),
                                        torch.flip(torch.LongTensor(interactions.user_id_contig.values), dims=[0])),
    })  # , device='cuda:0')

    #  1. Set graph edge features
    hg.edata['rating'] = {'rated': torch.LongTensor(interactions.rating.values)}  # .to('cuda:0')}
    hg.edata['rating'] = {'isRated': torch.flip(torch.LongTensor(interactions.rating.values), dims=[0])}  # .to('cuda:0')}

    # Set graph features

    # 2. Create train/test data with masks '''
    hg.nodes['user'].data['train_mask'] = torch.zeros(hg.num_nodes('user'),
                                                      dtype=torch.bool).bernoulli(0.7)  # .to(hg.device)
    hg.nodes['recipe'].data['train_mask'] = torch.zeros(hg.num_nodes('recipe'),
                                                        dtype=torch.bool).bernoulli(0.7)  # .to(hg.device)

    hg.edges['rated'].data['train_mask'] = torch.zeros(hg.num_edges('rated'),
                                                       dtype=torch.bool).bernoulli(0.7)  # .to(hg.device)

    # 3. Create embeddings for ingredient tokens in every recipe
    token_documents = [TaggedDocument(row[1]['ingredients'],
                                      [row[1]['recipe_id_contig']])
                       for row in recipes.sort_values(by=['recipe_id_contig']).iterrows()]
    token_model = Doc2Vec(token_documents, vector_size=70, window=2, min_count=1, workers=4)

    hg.ndata['ingredients'] = {'recipe': torch.from_numpy(token_model.dv.vectors).float()}  # .to('cuda:0')}

    # 4. Name representation: We're using doc2vec
    name_documents = [TaggedDocument(simple_preprocess(str(row[1]['name'])),
                                     [row[1]['recipe_id_contig']])
                      for row in recipes.sort_values(by=['recipe_id_contig']).iterrows()]
    name_model = Doc2Vec(name_documents, vector_size=20, window=2, min_count=1, workers=4)

    hg.ndata['name'] = {'recipe': torch.from_numpy(name_model.dv.vectors).float()}  # .to('cuda:0')}

    # User features are trainable embeddings
    embed_dict = {'user': torch.nn.Parameter(torch.FloatTensor(hg.num_nodes('user'), 70))}  # .to('cuda:0'))}
    torch.nn.init.xavier_uniform_(embed_dict['user'])
    hg.ndata['feat'] = {'user': embed_dict['user']}

    # save graph
    save_graphs('./data/data.bin', [hg])

    return hg
