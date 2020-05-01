import random
import logging

from eda import load_in_data

logging.basicConfig(level=logging.INFO)


def remove_nan_columns(data):
    cols_to_drop = ['visitor_hist_starrating', 'visitor_hist_adr_usd', 'srch_query_affinity_score',
                    'comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp2_rate_percent_diff',
                    'comp3_rate_percent_diff', 'comp4_rate_percent_diff', 'comp5_rate_percent_diff',
                    'comp6_rate_percent_diff', 'comp7_rate_percent_diff', 'comp8_rate_percent_diff', 'comp2_rate',
                    'comp3_rate', 'comp4_rate', 'comp5_rate', 'comp6_rate', 'comp7_rate', 'comp8_rate', 'comp2_inv',
                    'comp3_inv', 'comp4_inv', 'comp5_inv', 'comp6_inv', 'comp7_inv', 'comp8_inv', 'gross_bookings_usd']
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
    train_data_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
    cleaned_filepath = '../VU_DMT_assignment2/cleaned_training_data.csv'

    data = load_in_data(train_data_filepath)
    data = remove_nan_columns(data)
    impute_nan_values(data, cleaned_filepath)


if __name__ == "__main__":
    run()
