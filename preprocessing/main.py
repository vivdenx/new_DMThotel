import logging
import pickle

import convert_dates
from aggregate_comp_cols import aggregate_comp_cols
from down_sampling import down_sampling
import impute_nan_values

from eda import load_in_data

logging.basicConfig(level=logging.INFO)

# Define necessary filepaths.
original_filepath = '../../VU_DMT_assignment2/training_set_VU_DM.csv'
cleaned_filepath = '../../VU_DMT_assignment2/cleaned_training_data.csv'
resampled_filepath = '../../VU_DMT_assignment2/resampled_training_data.csv'
complete_filepath = '../../VU_DMT_assignment2/complete_training_data.csv'


def drop_columns(data):
    """
    Drop columns that have over 90% missing values. These features have been defined previously in EDA
        and have been saved as a pickle file in './reuseables/features_to_delete.pkl.

    :param data: Pandas DataFrame with the original data.
    :return: data (Pandas DataFrame with the 90% missing values removed).
    """
    # Load in Pickle file.
    feats2delete = pickle.load(open('reuseables/feats_to_delete.pkl', 'rb'))

    # Loop over the columns defined in the Pickle file and drop each from the dataset.
    for col in feats2delete:
        data.pop(col)

    filter_cols = [col for col in data if col.startswith('comp')]

    for col in filter_cols:
        data.pop(col)

    logging.info(
        f'The following features have been removed from the original dataset: {feats2delete} and {filter_cols}.')

    return data


def main():
    """
    Pre-processing steps:
    1. Add aggregated competitors column.
    2. Delete 90% missing columns and remaining comp_X columns.
    3. Down-sample data.
    4. Convert datetime column.
    5. Impute missing values for prop_review_score, prop_location_score2, and orig_destination_distance.
    6. Check!

    :return:
    """
    #data = load_in_data(original_filepath)
    #data = aggregate_comp_cols(data)
    #data = drop_columns(data)

    #resampled_data = down_sampling(data, resampled_filepath)

    resampled_data = load_in_data(resampled_filepath)

    resampled_data = convert_dates.convert_datetime(resampled_data)
    imputed_data = impute_nan_values.impute_nan_values(resampled_data)

    imputed_data.to_csv(complete_filepath, index=False)


if __name__ == "__main__":
    main()
