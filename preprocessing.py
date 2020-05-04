import logging
import pickle
import random

import pandas as pd

from eda import load_in_data

logging.basicConfig(level=logging.INFO)

train_data_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
cleaned_filepath = '../VU_DMT_assignment2/cleaned_training_data.csv'

feats_to_delete_path = 'pickles/feats_to_delete.pkl'


def date_col_to_datetime(data):
    """
    Convert the date_time column in the dataframe to datetime format.

    :param data: Pandas DataFrame that contains the entire dataset
    :return: data (Pandas DataFrame with the converted date_time column)
    """
    data['date_time'] = pd.to_datetime(date['date_time'])
    logging.info(f'Converted date_time column in data to datetime format.')
    return data


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


def impute_nan_values(df2, cleaned_filepath):
    """
    Randomise missing data for DataFrame (within a column)
    :param df2:
    :param cleaned_filepath:
    :return:
    """
    df = df2.copy()
    for col in df.columns:
        data = df['prop_review_score']
        mask = data.isnull()
        samples = random.choices(data[~mask].values, k=mask.sum())
        data[mask] = samples

    df['prop_location_score2'].fillna((df['prop_location_score2'].mean()), inplace=True)

    df['orig_destination_distance'].fillna((df['orig_destination_distance'].median()), inplace=True)
    df.to_csv(cleaned_filepath)


def run():
    data = load_in_data(train_data_filepath)
    data = date_col_to_datetime(data)
    data = remove_nan_columns(data)
    impute_nan_values(data, cleaned_filepath)


if __name__ == "__main__":
    run()
