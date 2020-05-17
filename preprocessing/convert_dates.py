import logging
from datetime import datetime

import pandas as pd

logging.basicConfig(level=logging.INFO)


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


def get_weekday(x):
    if x is not None and type(x) is not float:
        try:
            return datetime.strptime(x, '%Y-%m-%d').weekday()
        except:
            return datetime.strptime(x, '%Y-%m-%d %H:%M:%S').weekday()
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
    """
    Get the time of day of the search (1 for morning, 2 for afternoon, 3 for evening and 4 for night).

    :return: timeblock (which indicates the time of day)
    """
    x = get_hour(x)
    if x is not None and type(x) is not float:
        if 6 <= x and x < 12:
            timeblock = '1'
        elif 12 <= x and x < 18:
            timeblock = '2'
        elif 18 <= x and x < 22:
            timeblock = '3'
        elif 22 <= x or x < 6:
            timeblock = '4'
        return timeblock
    else:
        return 1
    pass


def left_merge_dataset(left_dframe, right_dframe, merge_column):
    return pd.merge(left_dframe, right_dframe, on=merge_column, how='left')


def convert_datetime(df):
    """
    Convert the date_time column in the dataframe to four different columns. One with the year, one with the month,
        one with the hour of the day and one with the time of day.

    :param df: Pandas DataFrame that contains the entire dataset
    :return: data (Pandas DataFrame with the converted date_time column)
    """
    df['year'] = pd.Series(df.date_time, index=df.index)
    df['month'] = pd.Series(df.date_time, index=df.index)
    df['weekday'] = pd.Series(df.date_time, index=df.index)
    df['hour'] = pd.Series(df.date_time, index=df.index)
    df['time_of_day'] = pd.Series(df.date_time, index=df.index)

    df.year = df.year.apply(lambda x: get_year(x))
    df.month = df.month.apply(lambda x: get_month(x))
    df.weekday = df.weekday.apply(lambda x: get_weekday(x))
    df.hour = df.hour.apply(lambda x: get_hour(x))
    df.time_of_day = df.time_of_day.apply(lambda x: get_timeblock(x))

    del df['date_time']
    logging.info('Converted date_time column into four different columns (year, month, hour and time of day).')

    return df
