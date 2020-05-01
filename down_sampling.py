import logging

import numpy as np
import pandas as pd

from eda import load_in_data

logging.basicConfig(level=logging.INFO)


def negative_down_sampling(cleaned_data_filepath, resampled_out_filepath):
    """
    Negative down sampling is used to reduce data size and create large sets of features without quality lost.
        In this case, it is used to balance the majority class (no-click) and the important minority class (click),
        which was heavily underrepresented in the original dataset.

    :param cleaned_data_filepath: path to the pre-processed dataset.
    :param resampled_out_filepath: path to which the resampled dataset will be written.
    """
    # Load in the pre-processed dataset.
    data = load_in_data(cleaned_data_filepath)

    # Find all indexes of the click data and create a random sample of the same length.
    click_indices = data[data.click_bool == 1].index
    random_indices = np.random.choice(click_indices, len(data.loc[data.click_bool == 1]), replace=False)
    click_sample = data.loc[random_indices]

    # Find all indexes of the non-click data and
    # create a random sample of the same length as the click data to balance out the dataset.
    not_click = data[data.click_bool == 0].index
    random_indices = np.random.choice(not_click, sum(data['click_bool']), replace=False)
    not_click_sample = data.loc[random_indices]

    # Combine the click and non-click samples.
    data_new = pd.concat([not_click_sample, click_sample], axis=0)

    # Get the percentages of click and non-click.
    percentage_non_click = len(data_new[data_new.click_bool == 0]) / len(data_new)
    percentage_click = len(data_new[data_new.click_bool == 1]) / len(data_new)

    logging.info(f"Percentage of non-click impressions: {percentage_non_click}.\n"
                 f"Percentage of click impressions: {percentage_click}.\n"
                 f"Total number of records in resampled data: {len(data_new)}.")

    # Write the resampled file to a new file.
    data_new.to_csv(resampled_out_filepath)
    logging.info(f'Resampled training data written to {resampled_out_filepath}.')


def run():
    # Define inpath to pre-processed data and outpath for the resampled data.
    cleaned_data_path = '../VU_DMT_assignment2/cleaned_training_data.csv'
    resampled_out_filepath = '../VU_DMT_assignment2/resampled_training_data.csv'

    negative_down_sampling(cleaned_data_path, resampled_out_filepath)


if __name__ == "__main__":
    run()
