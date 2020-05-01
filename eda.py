import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)


def load_in_data(training_data_filepath):
    data = pd.read_csv(training_data_filepath)
    return data


def missing_values_heatmap(data):  # TODO: Make sure it fits the image.
    ax = sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    fig = ax.get_figure()
    fig.savefig('./figures/missing_values.png')


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
    original_filepath = '../VU_DMT_assignment2/training_set_VU_DM.csv'
    cleaned_training_filepath = '../VU_DMT_assignment2/cleaned_training_data.csv'
    data = load_in_data(original_filepath)
    missing_values_heatmap(data)

    # data = load_in_data(cleaned_training_filepath)
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
