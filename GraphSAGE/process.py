import ast

import pandas as pd
import numpy as np


# function to remove items in list not in another given list
def filter_list(list_to_filter: list, set_to_check: frozenset):
    return list(filter(lambda item: item in set_to_check, list_to_filter))


def get_recipe_data(path: str):
    recipe_data = pd.read_csv(path)
    assert(len(recipe_data) > 0)

    # drop any rows where column value is nan
    recipe_data = recipe_data[~recipe_data.isnull().any(axis=1)]

    recipe_data = recipe_data.rename(columns={'id': 'recipe_id', 'contributor_id': 'user_id'})
    recipe_data = recipe_data.dropna()

    recipe_data['ingredients'] = recipe_data.ingredients.apply(ast.literal_eval)
    print("Recipe data: \n{}".format(recipe_data.info()))

    return recipe_data


def get_interactions_data(path: str):
    interactions_data = pd.read_csv(path)
    assert(len(interactions_data) > 0)

    # drop any rows where any column value is nan
    interactions_data = interactions_data[~interactions_data.isnull().any(axis=1)]
    print("Interactions data: \n{}".format(interactions_data.info()))
    return interactions_data


def clean_data(interactions, recipes):
    # Map user ids to contiguous numbers for heterograph creation
    _id = 0
    all_users = set(np.concatenate((interactions.user_id.unique(), recipes.user_id.unique())))

    user_id_to_contig_num_map = {}
    contig_id_to_user_id_map = {}
    for user in all_users:
        user_id_to_contig_num_map[user] = _id
        contig_id_to_user_id_map[_id] = user

        _id = _id + 1

    # Assign contiguous IDs to dataframe
    recipes['user_id_contig'] = recipes.user_id.apply(lambda x: user_id_to_contig_num_map[x])
    interactions['user_id_contig'] = interactions.user_id.apply(lambda x: user_id_to_contig_num_map[x])

    # Create a similar mapping for recipes
    r_id=0
    all_recipes = set(np.concatenate((recipes.recipe_id.unique(), interactions.recipe_id.unique())))

    recipe_id_to_contig_map = {}
    contig_id_to_recipe_id_map = {}

    for recipe in all_recipes:
        recipe_id_to_contig_map[recipe] = r_id
        contig_id_to_recipe_id_map[r_id] = recipe

        r_id = r_id + 1

    ''' Assign contiguous IDs '''
    recipes['recipe_id_contig'] = recipes['recipe_id'].apply(lambda x: recipe_id_to_contig_map[x])
    interactions['recipe_id_contig'] = interactions['recipe_id'].apply(lambda x: recipe_id_to_contig_map[x])

    # Sort interactions by recipe id
    interactions = interactions.sort_values(by=['user_id_contig'])

    # sort recipes by user_id
    recipes = recipes.sort_values(by=['recipe_id_contig'])

    return interactions, recipes


