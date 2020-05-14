import logging
import pickle
import random
from datetime import datetime
import numpy as np

import pandas as pd

from eda import load_in_data

logging.basicConfig(level=logging.INFO)

train_data_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
cleaned_filepath = '../VU_DMT_assignment2/cleaned_training_data.csv'
resampled_filepath = '../VU_DMT_assignment2/resampled_training_data.csv'
complete_filepath = '../VU_DMT_assignment2/complete_training_data.csv'

feats_to_delete_path = 'reuseables/feats_to_delete.pkl'


def get_year(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013
    pass


def get_month(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').month
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1
    pass


def left_merge_dataset(left_dframe, right_dframe, merge_column):
    return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')


def convert_datetime(df):
    """
    Convert the date_time column in the dataframe to datetime format.

    :param df: Pandas DataFrame that contains the entire dataset
    :return: data (Pandas DataFrame with the converted date_time column)
    """

    df['date_time_year'] = pd.Series(df.date_time, index=df.index)
    df['date_time_month'] = pd.Series(df.date_time, index=df.index)

    df.date_time_year = df.date_time_year.apply(lambda x: get_year(x))
    df.date_time_month = df.date_time_month.apply(lambda x: get_month(x))
    del df['date_time']
    logging.info('Converted date_time column into date_time format.')
    return df


def remove_nan_columns(data, feats_to_delete_path):
    """
    Remove all columns with over 90% missing data, as explored previously in the EDA.

    :param data: Pandas DataFrame which contains the entire dataset.
    :param feats_to_delete_path: path to a pickle file (.pkl) that contains a list of features with over 90%
        missing values, which are to be removed
    :return: data (Pandas DataFrame with columns removed)
    """
    # Load in the list from the pickle file.
    with open(feats_to_delete_path, 'rb') as f:
        cols_to_drop = pickle.load(f)

    # Remove the columns.
    data.drop(cols_to_drop, axis=1, inplace=True)
    logging.info(f"Removed the following columns: {cols_to_drop}.")

    return data


# TODO: aggregate and remove the comp_columns

# TODO Jupyter code

def impute_nan_values(resampled_df):
    # Load in DataFrames dictionaries for replacing NaN values
    review_df = load_in_data('./reuseables/prop_review_df.csv')
    loc_df = load_in_data('./reuseables/prop_location_score2.csv')
    orig_df = load_in_data('./reuseables/orig_destination_distance.csv')

    print(resampled_df["prop_location_score2"].isnull().sum())
    empty_indices = np.where(pd.isnull(resampled_df["prop_location_score2"]))

    for i in empty_indices:
        row = resampled_df["prop_id"].iloc[i]
        for ix, prop_id in row.items():
            value = loc_df.loc[review_df["prop_id"] == prop_id, "prop_location_score2"]
            resampled_df.loc[(resampled_df["prop_id"] == prop_id) & (resampled_df["prop_location_score2"].isnull()), "prop_location_score2"] =
            # print(resampled_df["prop_location_score2"].loc[resampled_df["prop_id"] == prop_id].fillna(value, inplace=True))
            # print(resampled_df["prop_review_score"])#.loc[resampled_df["prop_id"]==prop_id])
            # resampled_df["prop_review_score"] = resampled_df["prop_review_score"].loc[resampled_df["prop_id"] == prop_id].fillna(value, inplace=True)

    print(resampled_df["prop_location_score2"].isnull().sum())

    return resampled_df


def run():
    # data = load_in_data(train_data_filepath)
    # data = convert_datetime(data)
    # data = remove_nan_columns(data, feats_to_delete_path)
    # data.to_csv(cleaned_filepath)

    # filter_df = data[["prop_id", "srch_id", "prop_location_score2", "prop_review_score", "orig_destination_distance"]]

    # save_columns(filter_df)

    resampled_data = load_in_data(resampled_filepath)
    resampled_data = impute_nan_values(resampled_data)
    resampled_data.to_csv(complete_filepath)


if __name__ == "__main__":
    run()
