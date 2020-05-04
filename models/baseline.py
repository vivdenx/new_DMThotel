from eda import load_in_data

cleaned_training_filepath = "../../VU_DMT_assignment2/cleaned_training_data.csv"  # TODO: change to final cleaned datapath


def get_most_common_prop_id(data):  # TODO: ytrain
    most_common_prop_id = list(data.prop_id.value_counts().head().index)
    predictions = [most_common_prop_id for i in range(data.shape[0])]
    return most_common_prop_id


def run():
    data = load_in_data(cleaned_training_filepath)
    get_most_common_prop_id(data)


if __name__ == "__main__":
    run()
