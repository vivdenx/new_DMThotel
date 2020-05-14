import numpy as np
import pandas as pd

from eda import load_in_data


def get_empty_indices(df, t):
    # Find all empty indices for col1 and convert them to a list.
    filter_col, missing_col = t
    empty_indices = np.where(pd.isna(df[missing_col]))
    for i in empty_indices:
        empty = df.iloc[i][filter_col].to_list()

    return empty


def replace_missing_values(df, translation, empty, t):
    filter_col, missing_col = t

    for i in empty:
        value = translation.loc[translation[filter_col] == i, missing_col].item()
        df.loc[df[filter_col] == i, missing_col] = df.loc[df[filter_col] == i, missing_col].replace(np.nan, value)

    assert df[missing_col].isna().sum() == 0
    return df


def impute_nan_values(df):
    # Load in DataFrames dictionaries for replacing NaN values
    review_df = load_in_data('./reuseables/prop_review_df.csv')
    loc_df = load_in_data('./reuseables/prop_location_score2.csv')
    orig_df = load_in_data('./reuseables/orig_destination_distance.csv')

    # Define tuples with col1 (the column that will be filtered on) and col2 (the column with missing values).
    review_t = ("prop_id", "prop_review_score")
    loc_t = ("prop_id", "prop_location_score2")
    orig_t = ("srch_id", "orig_destination_distance")

    # Load in a list of empty indices for the column with missing values.
    review_empty = get_empty_indices(df, review_t)
    loc_empty = get_empty_indices(df, loc_t)
    orig_empty = get_empty_indices(df, orig_t)

    # Replace missing values
    df = replace_missing_values(df, review_df, review_empty, review_t)
    df = replace_missing_values(df, loc_df, loc_empty, loc_t)
    df = replace_missing_values(df, orig_df, orig_empty, orig_t)

    return df
