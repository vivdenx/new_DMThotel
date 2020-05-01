def get_most_common_prop_id(data):
    most_common_prop_id = list(data.hotel_cluster.value_counts().head())
    return most_common_prop_id