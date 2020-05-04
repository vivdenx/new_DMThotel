import logging
import pickle

import matplotlib.pyplot as plt
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


def get_missing_values(complete_df):
    """
    Find all features in the dataset with missing values, and the percentage of missing values.
    Draw a bar graph that shows these missing values.
    Log a list of missing values with over 90% missing that will be used later to remove columns.

    :param complete_df: pandas DataFrame with the entire dataset.
    """
    # Find the features with missing values.
    missing = complete_df.isnull().sum().to_frame()
    missing = missing.loc[missing[0] != 0]
    missing["percentage"] = (missing[0] / complete_df.shape[0] * 100)
    missing.sort_values(by=["percentage"])
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
    plt.savefig('./figures/EDA_missing_values.png')

    # Find which features contain over 90% missing values, which will be removed later.
    delete = missing.loc[missing["percentage"] > 90]
    feats_to_delete = delete.index.to_list()
    logging.info(f"These features contain over 90% missing data: {feats_to_delete}.")
    with open('pickles/feats_to_delete.pkl', 'wb') as f:
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

    # Plot a line graph that shows the distribution of the property IDs over the dataset.
    plt.plot(numbers, color='black', zorder=3)
    plt.grid(color='grey', linestyle='dashed', alpha=0.5, zorder=0)
    plt.xlabel('Number of properties')
    plt.ylabel('Number of times appeared in the dataset')
    plt.title('Property IDs in the training dataset')
    plt.tight_layout()
    plt.savefig('./figures/EDA_prop_ids.png')


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


def check_click_bool(data):
    sns.countplot(x='click_bool', data=data, palette='hls')
    plt.savefig('./figures/bar_click_bool.png')
    click_counts = data['click_bool'].value_counts()
    click_rate = click_counts[1] / (click_counts[0] + click_counts[1])
    click_rate = round((click_rate * 100), 1)
    logging.info(f'The click rate is {click_rate}%.')


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
    plt.savefig('./figures/heatmap.png')


# Alter filepath depending on where you have the data stored.
# original_training_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
# data = load_in_data(original_training_filepath)
# explore_country_figures(data)
def run():
    # Load in the data for training and test.
    # training_data = load_in_data(training_filepath)
    # test_data = load_in_data(test_filepath)

    # get_missing_values(training_data)

    # Load in data after pre-processing and continue EDA
    cleaned_training_data = load_in_data(cleaned_training_filepath)

    explore_prop_id(cleaned_training_data)

    # check_click_bool(data)
    # search_length_of_stay(data)
    # search_adults_counts(data)
    # search_property_star_rating(data)
    # check_property_brand(data)
    # check_stay_sat(data)
    # check_price(data)
    # get_heatmap(data)


if __name__ == "__main__":
    run()
