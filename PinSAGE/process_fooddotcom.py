"""
Script that reads from raw Food.com data and dumps into a pickle
file a heterogeneous graph with categorical and numeric features.
"""

import ast
import os
from operator import itemgetter
import argparse
from numpy.lib.function_base import sort_complex
import pandas as pd
import scipy.sparse as ssp
import pickle
from data_utils import *
from builder import PandasGraphBuilder

import matplotlib.pyplot as plt
import seaborn as sns
import pygraphviz as pgv

# See if I can get a bipartite graph visualization

# https://www.kaggle.com/ashwinik/siamese-bert-recipe-embeddings-recommendations 
# https://www.kaggle.com/zinayida/data-exploration-feature-engineering-pipeline

def avoidRowsWithMissValues(df):
    if(df.isnull().values.any()): 
        columns = df.columns
        for column in columns: 
            df[df[column].isnull()] = ""
            df[df[column]=='NaN'] = ""
            df[pd.isna(df[column])] = ""
    return df

def visualize_data(users, recipes, interactions):
    plt.close('all')

    plt.figure(1)
    recipes['n_steps'].value_counts().sort_index(ascending=True).plot(
        kind='bar', xlabel='Number of Steps', ylabel = 'Occurrences',
        title='Number of Steps in Recipes', figsize=(15,9))
    plt.savefig(os.path.join(output_path, 'Number_of_Steps_in_Recipes.png'))

    plt.figure(2)
    recipes['n_ingredients'].value_counts().sort_index(ascending=True).plot(
        kind='bar', xlabel='Number of Ingredients', ylabel = 'Occurrences',
        title='Number of Ingredients in Recipes', figsize=(15,9))
    plt.savefig(os.path.join(output_path, 'Number_of_Ingredients_in_Recipes.png'))

    plt.figure(3)
    interactions.groupby('recipe_id')['rating'].mean().reset_index().rating.plot(
        kind='hist', title='Histogram of Average Recipe Rating', 
        xlabel='Average Ratings', ylabel='Number of Recipes')
    plt.savefig(os.path.join(output_path, 'Histogram_of_Average_Recipe_Rating.png'))

    plt.figure(4)
    ax = recipes[recipes['minutes'] < 2_000_000].minutes.value_counts().sort_index(ascending=True).plot(
        title='Distribution of Recipe Minutes', xlabel='Minutes', ylabel='Occurrences')
    # ax.set_xscale('log')
    plt.savefig(os.path.join(output_path, 'Distribution_of_Recipe_Minutes.png'))
    

    plt.show()

    # ratings = pd.DataFrame(interactions.groupby('recipe_id').mean()['rating'])

    # sns.set(style = 'whitegrid')
    # ax = sns.boxenplot(x=ratings['rating'])
    # ax.set_xticks(np.arange(0, 6))
    # ax.set_xlabel('Ratings')
    # plt.show()

    # sns.set(style='whitegrid')
    # ax = sns.boxenplot(x = ratings['n_ingredients'])
    # ax.set_xticks(np.arange(0,20))
    # ax.set_xlabel('Number of ingredients per recipe')
    # plt.show()

    # sns.set(style = 'whitegrid')
    # ax = sns.boxenplot(x = ratings['n_steps'])
    # ax.set_xticks(np.arange(0,40, 2))
    # ax.set_xlabel('Number of steps per recipe')
    # plt.show()

    # Create a figure and axis
    #fig, ax = plt.subplots()

def visualize_graph(graph):
    # https://docs.dgl.ai/en/0.6.x/tutorials/basics/5_hetero.html
    ag = pgv.AGraph(strict=False, directed=True, landscape='true', ranksep='0.1')
    for u, v, k in graph.edges(form='all', etype='interacted'):
        ag.add_edge(u, v, label=k)
    for u, v, k in graph.edges(form='all', etype='interaction-by'):
        ag.add_edge(u, v, label=k)
    for u, v, k in graph.edges(form='all', etype='submitted'):
        ag.add_edge(u, v, label=k)
    for u, v, k in graph.edges(form='all', etype='submitted-by'):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('fooddotcom_graph.png')

def load_data_files(use_pp_files = False):
    if use_pp_files:
        # Read preprocessed interactions file
        print('Loading preprocessed interactions data...')
        interactions = pd.read_csv(os.path.join(directory, 'pp_interactions.csv'))
        
        # Read preprocessed recipes file
        print('Loading preprocessed recipes data...')
        recipes = pd.read_csv(os.path.join(directory, 'pp_recipes.csv'))

        # Read preprocessed recipes file
        print('Loading preprocessed user data...')
        users = pd.read_csv(os.path.join(directory, 'pp_users.csv'))
    else:
        # Read raw interactions file
        print('Loading raw interactions data...')
        interactions = pd.read_csv(os.path.join(directory, 'RAW_interactions.csv'))
        
        # Read raw recipes file
        print('Loading raw recipes data...')
        recipes = pd.read_csv(os.path.join(directory, 'RAW_recipes.csv'))

        # Rename recipe id column
        print('Renaming recipe "id" column to "recipe_id"...')
        recipes.rename(columns={'id':'recipe_id'}, inplace=True)

        # Output statistics on each raw dataframe to CSV file
        print('Creating raw recipes table description file...')
        recipes.describe(include="all").to_csv(os.path.join(output_path, 'raw_recipes_describe.csv'))
        
        print('Creating raw interactions table description file...')
        interactions.describe(include="all").to_csv(os.path.join(output_path, 'raw_interactions_describe.csv'))

        # Drop NaN/None/Null values
        print('Dropping NaN/None/Null values...')
        recipes = recipes[~recipes.isnull().any(axis=1)]
        interactions = interactions[~interactions.isnull().any(axis=1)]

        # Drop bad minutes value
        print('Dropping minutes values greater than 95th percentile...')
        recipes = recipes[recipes['minutes'] <= recipes['minutes'].quantile(0.95)]

        counter = 0
        print('Parsing out users and recipes not shared between interaction and recipe tables...')
        while (not recipes['contributor_id'].isin(interactions['user_id']).all()
                or not recipes['recipe_id'].isin(interactions['recipe_id']).all()
                or not interactions['recipe_id'].isin(recipes['recipe_id']).all()):

            counter += 1

            # Drop recipes with users that aren't in interactions
            print(f'[{counter}] Removing recipes with users not in interactions table...')
            recipes = recipes[recipes['contributor_id'].isin(interactions['user_id'])]

            # Drop recipes that aren't in interactions table
            print(f'[{counter}] Removing recipes with ids not in interactions table...')
            recipes = recipes[recipes['recipe_id'].isin(interactions['recipe_id'])]

            # Drop interactions with recipes that aren't in recipes table
            print(f'[{counter}] Removing interactions with recipes not in recipes table...')
            interactions = interactions[interactions['recipe_id'].isin(recipes['recipe_id'])]

            print(f'[{counter}] interactions shape: {interactions.shape}')
            print(f'[{counter}] recipe shape: {recipes.shape}')

        # Remap user ids into contiguous ids (recipes has removed all
        # users not in interactions, so just use interactions)
        print('Creating contiguous user id mapping...')
        user_ids = set(interactions['user_id'].tolist())
        user_ids = sorted(user_ids)
        user_id_contiguous_map = {val:index for index, val in enumerate(user_ids)}

        # Remap recipe ids into contiguous ids (interactions has removed
        # all recipes not in recipe table, so just use recipes)
        print('Creating contiguous recipe id mapping...')
        recipe_ids = set(recipes['recipe_id'].tolist())
        recipe_ids = sorted(recipe_ids)
        recipe_id_contiguous_map = {val:index for index, val in enumerate(recipe_ids)}

        # Create mapping functions
        user_id_mapping = lambda x : user_id_contiguous_map[x]
        recipe_id_mapping = lambda x : recipe_id_contiguous_map[x]

        # Create contiguous recipe and user id columns in recipes
        print('Creating contiguous user (u) and recipe (i) id columns in recipes table...')
        recipes['u'] = recipes.contributor_id.map(user_id_mapping)
        recipes['i'] = recipes.recipe_id.map(recipe_id_mapping)

        # Create contiguous recipe and user id columns in interactions
        print('Creating contiguous user (u) and recipe (i) id columns in interactions table...')
        interactions['u'] = interactions.user_id.map(user_id_mapping)
        interactions['i'] = interactions.recipe_id.map(recipe_id_mapping)

        # Get number of users and recipes left
        num_users = len(user_ids)
        num_recipes = len(recipe_ids)
        print(f'Number of users: {num_users}')
        print(f'Number of recipes: {num_recipes}')

        # Create users table
        u_u = []
        u_user_id = []
        u_items = []
        u_n_items = []
        u_submissions = []
        u_n_submissions = []
        u_ratings = []
        u_n_ratings = []
        percent_threshold = int(num_users / 20)
        print('Creating user data lists [', end='')
        for u_id in range(num_users):
            u_u.append(u_id)
            u_user_id.append(user_ids[u_id])
            try:
                i_df = interactions[interactions['u'] == u_id]
                items = i_df['i'].tolist()
                ratings = i_df['rating'].tolist()
            except:
                items = []
                ratings = []
            
            try:
                r_df = recipes[recipes['u'] == u_id]
                submissions = r_df['i'].tolist()
            except:
                submissions = []
            
            u_items.append(items)
            u_n_items.append(len(items))
            u_submissions.append(submissions)
            u_n_submissions.append(len(items))
            u_ratings.append(ratings)
            u_n_ratings.append(len(ratings))
            if u_id % percent_threshold == 0:
                print('.', end='')
        print(']')

        # Create users dataframe
        print('Creating users table...')
        user_data = {
            'u': u_u, 
            'user_id': u_user_id, 
            'items': u_items, 
            'n_items': u_n_items, 
            'submissions': u_submissions,
            'n_submissions': u_n_submissions,
            'ratings': u_ratings, 
            'n_ratings': u_n_ratings
        }
        users = pd.DataFrame.from_dict(user_data)
        print('User table completed!')

        # Delete large unused variables
        del recipe_ids
        del user_ids
        del u_u
        del u_user_id
        del u_items
        del u_n_items
        del u_submissions
        del u_n_submissions
        del u_ratings
        del u_n_ratings
        del user_data

        # Change column types
        print('Changing table column data types...')
        recipes['tags'] = recipes.tags.transform(ast.literal_eval)
        recipes['nutrition'] = recipes.nutrition.transform(ast.literal_eval)
        recipes['steps'] = recipes.steps.transform(ast.literal_eval)
        recipes['submitted'] = pd.to_datetime(recipes['submitted']).astype(np.int64)
        interactions['date'] = pd.to_datetime(interactions['date']).astype(np.int64)

        # Create nutrition columns
        print('Transforming recipes nutrition column into individual columns...')
        nutrition_columns = ['calories', 'total_fat', 'sugar', 'sodium',
                            'protein', 'saturated_fat', 'carbohydrates']
        recipes = recipes.join(recipes['nutrition'].transform({col_name: itemgetter(index) 
                                                for index, col_name 
                                                in enumerate(nutrition_columns)}))

        # Output preprocessed user dataframe to a CSV file
        print(f'[Writing all output files to {output_path}]')
        print('Writing preprocessed users table to pp_users.csv...')
        users.to_csv(os.path.join(output_path, 'pp_users.csv'))

        # Output preprocessed recipe dataframe to a CSV file
        print('Writing preprocessed recipes table to pp_recipes.csv...')
        recipes.to_csv(os.path.join(output_path, 'pp_recipes.csv'))

        # Output preprocessed interaction dataframe to a CSV file
        print('Writing preprocessed interactions table to pp_interactions.csv...')
        interactions.to_csv(os.path.join(output_path, 'pp_interactions.csv'))

    return interactions, recipes, users

def build_graph(interactions, recipes, users):
    # Create graph builder for PinSAGE input
    print('Building bipartite graph...')
    graph_builder = PandasGraphBuilder()

    # Sort dataframes
    users = users.sort_values(by='u')
    interactions = interactions.sort_values(by='u')
    recipes = recipes.sort_values(by='i')

    # Add node types and relations between them
    graph_builder.add_entities(users, 'u', 'user')
    graph_builder.add_entities(recipes, 'i', 'recipe')
    graph_builder.add_binary_relations(interactions, 'u', 'i', 'interacted')
    graph_builder.add_binary_relations(interactions, 'i', 'u', 'interaction-by')
    
    # Build the bipartite, heterogenous graph from users, recipes, and their interacitons
    g = graph_builder.build()
    print('Graph built!')

    # Add features to recipe nodes
    print('Adding features to recipe nodes...')
    # g.nodes['recipe'].data['calories'] = torch.FloatTensor(recipes['calories'].values)
    # g.nodes['recipe'].data['total_fat'] = torch.FloatTensor(recipes['total_fat'].values)
    # g.nodes['recipe'].data['sugar'] = torch.FloatTensor(recipes['sugar'].values)
    # g.nodes['recipe'].data['sodium'] = torch.FloatTensor(recipes['sodium'].values)
    # g.nodes['recipe'].data['protein'] = torch.FloatTensor(recipes['protein'].values)
    # g.nodes['recipe'].data['saturated_fat'] = torch.FloatTensor(recipes['saturated_fat'].values)
    # g.nodes['recipe'].data['carbohydrates'] = torch.FloatTensor(recipes['carbohydrates'].values)
    g.nodes['recipe'].data['minutes'] = torch.LongTensor(recipes['minutes'].values)
    g.nodes['recipe'].data['n_steps'] = torch.LongTensor(recipes['n_steps'].values)
    g.nodes['recipe'].data['n_ingredients'] = torch.LongTensor(recipes['n_ingredients'].values)
    
    
    # Add features to interaction edges
    print('Adding features to interaction edges...')
    g.edges['interacted'].data['rating'] = torch.LongTensor(interactions['rating'].values)
    g.edges['interacted'].data['date'] = torch.LongTensor(interactions['date'].values)
    g.edges['interaction-by'].data['rating'] = torch.LongTensor(interactions['rating'].values)
    g.edges['interaction-by'].data['date'] = torch.LongTensor(interactions['date'].values)
    
    print('All features added to graph!')

    return g

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('--user-csv', type=str, default=None)
    parser.add_argument('--expanded-recipes-csv', type=str, default=None)
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--load-preprocessed-files', action='store_true')
    args = parser.parse_args()
    directory = args.directory
    output_path = args.output_path
    visualization = args.visualization
    use_pp_files = args.load_preprocessed_files

    # Get preprocessed interactions, recipes, and users
    interactions, recipes, users = load_data_files(use_pp_files)

    # Build graph
    g = build_graph(interactions, recipes, users)

    # Create train-test split based on date
    print('Creating train-test split via the date submitted (e.g. to test on the future)...')
    train_indices, val_indices, test_indices = train_test_split_by_time(interactions, 'date', 'u')

    # Show indices sizes
    print(f'train_indices shape: {train_indices.shape}')
    print(f'val_indices shape:   {val_indices.shape}')
    print(f'test_indices shape:  {test_indices.shape}')

    # Build the graph with training interactions only.
    print('Building training graph...')
    train_g = build_train_graph(g, train_indices, 'user', 'recipe', 'interacted', 'interaction-by')

    # Remove users that are orphaned in training graph so that we can
    # evaluate successfully during training
    print('Removing orphaned users from training graph, graph, validation and test matrices...')
    bad_users = [x.item() for x in train_g.nodes(ntype='user') if train_g.out_degrees(x, etype='interacted') == 0]
    g.remove_nodes(bad_users, ntype='user')
    train_g.remove_nodes(bad_users, ntype='user')
    updated_interactions = interactions[~interactions['u'].isin(bad_users)]
    train_indices = updated_interactions['train_mask'].to_numpy().nonzero()[0]
    val_indices = updated_interactions['val_mask'].to_numpy().nonzero()[0]
    test_indices = updated_interactions['test_mask'].to_numpy().nonzero()[0]
    assert train_g.out_degrees(etype='interacted').min() > 0

    # Build the user-item sparse matrix for validation and test set.
    print('Building validation and test matrices...')
    val_matrix, test_matrix = build_val_test_matrix(g, val_indices, test_indices, 'user', 'recipe', 'interacted')
        
    # Create textual dataset for training
    fooddotcom_textual_dataset = {
        'name': recipes['name'].values.astype(str),
        # 'description': recipes['description'].values.astype(str),
        # 'review': interactions['review'].values.astype(str)
    }

    # Create dataset dictionary
    dataset = {
        'full-graph': g,
        'train-graph': train_g,
        'val-matrix': val_matrix,
        'test-matrix': test_matrix,
        'item-texts': fooddotcom_textual_dataset,
        'item-images': None,
        'user-type': 'user',
        'item-type': 'recipe',
        'user-to-item-type': 'interacted',
        'item-to-user-type': 'interaction-by',
        'timestamp-edge-column': 'date'}

    # Create pickle file of dataset
    print('Creating data pickle file of processed graphic dataset...')
    with open(os.path.join(output_path, 'food_dataset.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    

    if visualization:
        # Output statistics on each dataframe to CSV file
        print('Creating user table description file...')
        ud_df = users.describe(include="all")
        ud_df.to_csv(os.path.join(output_path, 'users_describe.csv'))
        
        print('Creating recipes table description file...')
        rd_df = recipes.describe(include="all")
        rd_df.to_csv(os.path.join(output_path, 'recipes_describe.csv'))
        
        print('Creating interactions table description file...')
        id_df = interactions.describe(include="all")
        id_df.to_csv(os.path.join(output_path, 'interactions_describe.csv'))

        # Get information about data shape
        print(f'Users table shape: {users.shape}')
        print(f'User table column names: {users.columns}')
        print(f'User table description:\n{ud_df}')
        print()

        print(f'Recipes table shape: {recipes.shape}')
        print(f'Recipes table column names: {recipes.columns}')
        print(f'Recipes table description:\n{rd_df}')
        print()

        print(f'Interactions table shape: {interactions.shape}')
        print(f'Interactions table column names: {interactions.columns}')
        print(f'Interactions table description:\n{id_df}')
        print()

        # Remove unnecessary description dataframes
        del ud_df
        del rd_df
        del id_df

        # Visualize data in plots
        print('Visualizing data...')
        visualize_data(users, recipes, interactions)
        # visualize_graph(g)

    print('Food.com dataset successfully processed!')