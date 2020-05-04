import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from eda import load_in_data


def get_ensemble_models():
    rf = RandomForestClassifier(n_estimators=51, min_samples_leaf=5, min_samples_split=3)
    bagg = BaggingClassifier(n_estimators=51, random_state=42)
    extra = ExtraTreesClassifier(n_estimators=51, random_state=42)
    ada = AdaBoostClassifier(n_estimators=51, random_state=42)
    grad = GradientBoostingClassifier(n_estimators=51, random_state=42)
    classifier_list = [rf, bagg, extra, ada, grad]
    classifier_name_list = ['Random Forests', 'Bagging', 'Extra Trees', 'AdaBoost', 'Gradient Boost']
    return classifier_list, classifier_name_list


def get_naive_bayes_models():
    gnb = GaussianNB()
    mnb = MultinomialNB()
    bnb = BernoulliNB()
    classifier_list = [gnb, mnb, bnb]
    classifier_name_list = ['Gaussian NB', 'Multinomial NB', 'Bernoulli NB']
    return classifier_list, classifier_name_list


def get_neural_network(hidden_layer_size=50):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_size)
    return [mlp], ['MultiLayer Perceptron']


def print_evaluation_metrics(trained_model, trained_model_name, X_test, y_test):
    print('--------- Model : ', trained_model_name, ' ---------------\n')
    predicted_values = trained_model.predict(X_test)
    print(metrics.classification_report(y_test, predicted_values))
    print("Accuracy Score : ", metrics.accuracy_score(y_test, predicted_values))
    print("---------------------------------------\n")


def get_training_split(training_data_filepath):
    data = load_in_data(training_data_filepath)
    y = data.booking_bool
    data.pop('click_bool')
    data.pop('booking_bool')

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    X_train.to_csv('data/training/X_data.csv')
    X_test.to_csv('data/dev/X_data.csv')
    y_train.to_csv('data/training/y_data.csv')
    y_test.to_csv('data/dev/y_data.csv')

    return X_train, X_test, y_train, y_test


def get_classifier(training_data_filepath, classifier_list, classifier_name_list):
    X_train, X_test, y_train, y_test = get_training_split(training_data_filepath)

    for classifier, classifier_name in zip(classifier_list, classifier_name_list):
        classifier.fit(X_train, y_train)
        print_evaluation_metrics(classifier, classifier_name, X_test, y_test)


def get_feature_importance():
    rf = RandomForestClassifier(n_estimators=51, min_samples_leaf=5, min_samples_split=3)
    rf.fit(X_train, y_train)
    (pd.Series(rf.feature_importances_, index=data.columns).plot(kind='barh'))


def run():
    training_data_filepath = "./cleaned_data/resampled_training_data.csv"

    classifier_list, classifier_name_list = get_ensemble_models()
    get_classifier(training_data_filepath, classifier_list, classifier_name_list)

    classifier_list, classifier_name_list = get_naive_bayes_models()
    get_classifier(training_data_filepath, classifier_list, classifier_name_list)

    classifier_list, classifier_name_list = get_neural_network()
    get_classifier(training_data_filepath, classifier_list, classifier_name_list)


if __name__ == "__main__":
    run()
