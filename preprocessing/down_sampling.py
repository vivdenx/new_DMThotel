import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def down_sampling(data, resampled_out_filepath):
    """
    Negative down sampling is used to reduce data size and create large sets of features without quality lost.
        In this case, it is used to balance the majority class (no-book) and the important minority class (book),
        which was heavily underrepresented in the original dataset.

    :param cleaned_data_filepath: path to the pre-processed dataset.
    :param resampled_out_filepath: path to which the resampled dataset will be written.
    """
    # Find all indexes of the book data and create a random sample of the same length.
    book_indices = data[data.booking_bool == 1].index
    random_indices = np.random.choice(book_indices, len(data.loc[data.booking_bool == 1]), replace=False)
    book_sample = data.loc[random_indices]

    # Find all indexes of the non-book data and
    # create a random sample of the same length as the book data to balance out the dataset.
    not_book = data[data.booking_bool == 0].index
    random_indices = np.random.choice(not_book, sum(data['booking_bool']), replace=False)
    not_book_sample = data.loc[random_indices]

    # Combine the book and non-book samples.
    data_new = pd.concat([not_book_sample, book_sample], axis=0)

    # Get the percentages of book and non-book.
    percentage_non_book = len(data_new[data_new.booking_bool == 0]) / len(data_new)
    percentage_book = len(data_new[data_new.booking_bool == 1]) / len(data_new)

    logging.info(f"Percentage of non-book impressions: {percentage_non_book}.\n"
                 f"Percentage of book impressions: {percentage_book}.\n"
                 f"Total number of records in resampled data: {len(data_new)}.")

    # Write the resampled file to a new file.
    data_new.to_csv(resampled_out_filepath, index=False)
    logging.info(f'Resampled training data written to {resampled_out_filepath}.')

    return data_new
