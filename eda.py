import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO)

training_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
test_filepath = '../VU_DMT_assignment2/test_set_VU_DM.csv'

cleaned_training_filepath = '../VU_DMT_assignment2/cleaned_training_data.csv'


def load_in_data(csv_data_filepath):
    """
    Function to load in the data (.csv format). Logs the shape of the DataFrame (rows, columns)
        and the number of columns as well as a list of all columns in the DataFrame

    :param csv_data_filepath: path to a .csv file that contains the data
    :return: data (pd.DataFrame)
    """
    data = pd.read_csv(csv_data_filepath)
    # data.info()
    return data


def get_missing_values(complete_df, eda_num):
    """
    Find all features in the dataset with missing values, and the percentage of missing values.
    Draw a bar graph that shows these missing values.
    Log a list of missing values with over 90% missing that will be used later to remove columns.

    :param complete_df: pandas DataFrame with the entire dataset.
    :param eda_num: value indicating whether it is before or after initial pre-processing
        (1 if it's the first run-through without any processing of the data;
        2 if it's the second run through after removing the initial empty values)
    """
    # Find the features with missing values.
    missing = complete_df.isnull().sum().to_frame()
    missing = missing.loc[missing[0] != 0]
    missing["percentage"] = (missing[0] / complete_df.shape[0] * 100)
    missing = missing.sort_values(by=["percentage"])
    features = missing.index.to_list()
    numbers = missing["percentage"].to_list()

    # Plot bar graph with missing value information.
    plt.bar(features, numbers, color='lightblue', zorder=3)
    plt.grid(color='grey', linestyle='dashed', alpha=0.5, zorder=0)
    plt.xlabel('Feature value')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage of entries missing')
    plt.title('Missing values per feature')
    plt.tight_layout()
    plt.savefig(f'./figures/EDA{eda_num}_missing_values.png')

    # Find which features contain over 90% missing values, which will be removed later.
    # This is only done for the first run over the missing values.
    if eda_num == 1:
        delete = missing.loc[missing["percentage"] > 90]
        feats_to_delete = delete.index.to_list()
        logging.info(f"These features contain over 90% missing data: {feats_to_delete}.")
        with open('preprocessing/reuseables/feats_to_delete.pkl', 'wb') as f:
            pickle.dump(feats_to_delete, f)


def explore_prop_id(cleaned_data):
    """
    Explore the property IDs in the cleaned training dataset. The top 5 common properties are logged,
        which can be used to develop a simple benchmark later.
        Plot a line graph that shows the distribution of the property ideas over the dataset.
        It's clear there's a Zipfian distribution of the data, with the most booked properties
        clustering around a few examples.

    :param cleaned_data: pandas DataFrame with the cleaned_dataset
    """
    # Load prop_id data from dataset.
    ids = cleaned_data.prop_id.value_counts().to_frame()
    logging.info(f"The 5 most common properties are {ids[:5]}.")
    numbers = ids['prop_id'].to_list()

    # Plot a line graph that shows the distribution of the property IDs over the dataset in a Ziphian curve.
    plt.plot(numbers, color='black', zorder=3)
    plt.grid(color='grey', linestyle='dashed', alpha=0.5, zorder=0)
    plt.xlabel('Number of properties')
    plt.ylabel('Number of times appeared in the dataset')
    plt.title('Property ID distribution in the training dataset')
    plt.tight_layout()
    plt.savefig('./figures/EDA_prop_ids.png')

    # Plot a graph that shows the distribution of the property IDs over the dataset.
    plt.figure(figsize=(12, 6))
    sns.distplot(cleaned_data['prop_id'])
    plt.xlabel('Property ID')
    plt.title('Property ID distribution in the training set')
    plt.tight_layout()
    plt.savefig('./figures/EDA_prop_id_plot.png')


def explore_country_figures(data):
    n, bins, patches = plt.hist(data.prop_country_id, 100, density=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Property country Id')
    plt.title('Histogram of prop_country_id')
    plt.savefig('./figures/EDA_prop_country_id.png')

    print(data.groupby('prop_country_id').size().nlargest(5))

    n, bins, patches = plt.hist(data.visitor_location_country_id, 100, density=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Visitor location country Id')
    plt.title('Histogram of visitor_location_country_id')
    plt.savefig('./figures/EDA_visitor_location_country_id.png')

    print(data.groupby('visitor_location_country_id').size().nlargest(5))


def search_length_of_stay(data):
    n, bins, patches = plt.hist(data.srch_length_of_stay, 50, density=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Search length of stay')
    plt.title('Histogram of search_length_of_stay')
    plt.axis([0, 30, 0, 0.65])
    plt.savefig('./figures/length_of_stay.png')
    stay_data = data.groupby('srch_length_of_stay').size().nlargest(5)
    print(stay_data)


def search_adults_counts(data):
    n, bins, patches = plt.hist(data.srch_adults_count, 20, density=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Search adults count')
    plt.title('Histogram of search_adults_count')
    plt.savefig('./figures/adults_counts.png')
    adult_counts_data = data.groupby('srch_adults_count').size().nlargest(5)
    print(adult_counts_data)


def search_property_star_rating(data):
    n, bins, patches = plt.hist(data.prop_starrating, 20, density=1, facecolor='blue', alpha=0.75)
    plt.xlabel('Property star rating')
    plt.title('Histogram of prop_star_rating')
    plt.savefig('./figures/property_star_rating.png')


def check_property_brand(data):
    brand_data = data.groupby('prop_brand_bool').size()
    brand_rate = brand_data[1] / (brand_data[0] + brand_data[1])
    brand_rate = round((brand_rate * 100), 1)
    logging.info(f'{brand_rate}% of the properties clicked on are brand properties.')


def check_stay_sat(data):
    sat_data = data.groupby('srch_saturday_night_bool').size()
    sat_rate = sat_data[1] / (sat_data[0] + sat_data[1])
    sat_rate = round((sat_rate * 100), 1)
    logging.info(f'{sat_rate}% of the properties click on are for stays on Saturday.')


def check_price(data):  # TODO: something goes massively wrong here
    sns.set(style="ticks", palette="pastel")

    ax = sns.boxplot(x="click_bool", y="price_usd", hue="click_bool", data=data)
    ax.set_ylim([0, 200])
    fig = ax.get_figure()
    fig.savefig('./figures/price_usd.png')

    print(data.groupby('click_bool')['price_usd'].describe())


def get_heatmap(data):
    data_corr = data.corr()
    mask = np.zeros_like(data_corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(data_corr.sort_values(by=['prop_id'], ascending=False), vmin=-1, vmax=1, center=0, cmap='coolwarm',
                     square=True, mask=mask)
    plt.tight_layout()
    plt.savefig('./figures/EDA_heatmap.png')


def check_click_booking_bool(data):
    # Get the total amount of clicks and divide it by the total dataset to get the percentage.
    click_counts = data['click_bool'].value_counts()
    click_rate = click_counts[1] / len(data['click_bool'])
    click_rate = round((click_rate * 100), 1)

    # Get the total amount of bookings and divide it by the total dataset to get the percentage.
    booking_counts = data['booking_bool'].value_counts()
    booking_rate = booking_counts[1] / len(data['booking_bool'])
    booking_rate = round((booking_rate * 100), 1)

    logging.info(f"The percentage of clicked properties is {click_rate}%.\n"
                 f"The percentage of booked properties is {booking_rate}%")

    x = ["click", "booking"]
    y = [click_rate, booking_rate]

    plt.bar(x, y)
    plt.ylabel('Percentage')
    plt.title('Percentage of clicked and booked properties in the dataset.')
    plt.tight_layout()
    plt.savefig('./figures/EDA_click_book_bool.png')


# Alter filepath depending on where you have the data stored.
# original_training_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
# data = load_in_data(original_training_filepath)
# explore_country_figures(data)
def eda1():
    # Load in the data for training and test.
    training_data = load_in_data(training_filepath)
    test_data = load_in_data(test_filepath)

    # get_missing_values(training_data, 1)
    print(training_data.corr()["prop_id"].sort_values())

    # Remove missing values in pre_processing.


def eda2():
    # Load in data after initial pre-processing and continue EDA.
    cleaned_training_data = load_in_data(cleaned_training_filepath)

    # get_missing_values(cleaned_training_data, 2)
    print(cleaned_training_data.corr()["prop_id"].sort_values())
    # explore_prop_id(cleaned_training_data)
    DATA = load_in_data('./data/new_cleaned_data_dates.csv')
    get_heatmap(DATA)
    # check_click_booking_bool(cleaned_training_data)


def run():
    eda1()
    eda2()


if __name__ == "__main__":
    run()
