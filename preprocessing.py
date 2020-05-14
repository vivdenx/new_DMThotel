import logging
import pickle
from datetime import datetime

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


def get_hour(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').hour
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').hour
    else:
        return 1
    pass


def get_timeblock(x):
    if x is not None and type(x) is not float:
        if 5 <= x and x < 12:
            timeblock = 'Mo'
        elif 12 <= x and x < 18:
            timeblock = 'Af'
        elif 18 <= x and x < 22:
            timeblock = 'Ev'
        elif 22 <= x or x < 5:
            timeblock = 'Ni'
        return timeblock
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
    df['date_time_hour'] = pd.Series(df.date_time, index=df.index)
    df['date_time_block'] = pd.Series(df.date_time_hour, index=df.index)

    df.date_time_year = df.date_time_year.apply(lambda x: get_year(x))
    df.date_time_month = df.date_time_month.apply(lambda x: get_month(x))
    df.date_time_hour = df.date_time_hour.apply(lambda x: get_hour(x))
    df.date_time_block = df.date_time_block.apply(lambda x: get_timeblock(x))
    del df['date_time']
    logging.info('Converted date_time column into date_time format.')
    return df


# TODO: aggregate and remove the comp_columns


def run():
    # data = load_in_data(train_data_filepath)
    # data = convert_datetime(data)
    # data = remove_nan_columns(data, feats_to_delete_path)
    # data.to_csv(cleaned_filepath)
    # save_columns(filter_df)

    resampled_data = load_in_data(resampled_filepath)
    impute_nan_values(resampled_data)


if __name__ == "__main__":
    run()
