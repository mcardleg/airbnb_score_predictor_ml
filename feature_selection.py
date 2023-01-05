import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def read_in_files():
    pre_normalized_numerical = pd.read_csv("numerical_data.csv")
    numerical = pd.DataFrame(
        data=normalize(X=pre_normalized_numerical, norm='max'), columns=pre_normalized_numerical.columns)
    categorical = pd.read_csv("ohe_listings.csv")
    listings_text = pd.read_csv("textual_data/CountVectorizer_90_1-5.csv")
    reviews_text = pd.read_csv("ohe_reviews.csv")
    pre_normalized_concatenated = pd.concat([numerical, categorical, listings_text, reviews_text], axis=1)

    return {
        "numerical": numerical,
        "categorical": categorical,
        "listings_text": listings_text,
        "reviews_text": reviews_text,
        "concatenated": pd.DataFrame(
            data=normalize(X=pre_normalized_concatenated, norm='max'), columns=pre_normalized_concatenated.columns)
    }


def cross_validate_for_c(dataset_name, X):
    y_df = pd.read_csv("../scores.csv")

    for index, score_type in enumerate(y_df.columns):
        print("     for " + score_type)
        kf = KFold(n_splits=5)
        mean_error = []
        std_error = []

        c_range = [0.1, 10, 100, 250, 500]
        for C in c_range:
            model = Lasso(alpha=1 / (2 * C), max_iter=100000)
            fold = []

            for train, test in kf.split(X):
                model.fit(X.iloc[train], y_df[score_type].iloc[train])
                predictions = model.predict(X.iloc[test])
                fold.append(mean_squared_error(y_df[score_type].iloc[test], predictions))

            mean_error.append(np.array(fold).mean())
            std_error.append(np.array(fold).std())

        plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3, label="Target field = {}".format(score_type))

    plt.rc('font', size=12)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.title("Cross validation for C for feature selection")
    plt.xlabel("C")
    plt.ylabel("Mean squared error")
    plt.legend()
    # plt.show()
    plt.savefig('crossval_{}.png'.format(dataset_name))


def get_useful_features(dataset_name, X):
    y_df = pd.read_csv("../scores.csv")
    weights_df = pd.DataFrame(columns=y_df.columns, index=X.columns)

    selected_c = {
        "numerical": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        "categorical": [100, 0.1, 100, 0.1, 0.1, 0.1],
        "listings_text": [100, 0.1, 100, 0.1, 0.1, 100],
        "reviews_text": [100, 100, 100, 100, 100, 100],
        "concatenated": [100, 100, 100, 0.1, 0.1, 0.1]
    }
    for index, score_type in enumerate(y_df.columns):
        print("     for " + score_type)
        y = y_df[score_type]
        model = Lasso(alpha=1 / (2 * selected_c.get(dataset_name)[index]))
        model.fit(X, y)
        weights_df[score_type] = model.coef_

    weights_df.to_csv("feature_engineering/weights/{}.csv".format(dataset_name))


def list_useful_fields(score, dataset_name, dataset):
    weights = pd.read_csv("feature_engineering/weights/{}.csv".format(dataset_name))
    selected_features = pd.DataFrame()

    column_names = []
    indices = weights[weights[score] != 0].index.tolist()
    for index in indices:
        column_names.append(weights.iat[index, 0])

    for column in column_names:
        selected_features[column] = dataset[column]
    selected_features.to_csv("feature_engineering/selected/{}_{}.csv".format(score, dataset_name), index=False)


def merge_final_datasets(score_type, datatypes):
    all_datatypes = []
    for datatype in datatypes:
        try:
            all_datatypes.append(pd.read_csv("feature_engineering/selected/{}_{}.csv".format(score_type, datatype)))
        except pd.errors.EmptyDataError:
            print("          {} csv was empty.".format(datatype))
            continue

    pd.concat(all_datatypes, axis=1).to_csv("engineered_data/{}_dataset.csv".format(score_type), index=False)


def feature_selection(y_fields):
    X = read_in_files()

    for dataset in X:
        print("Cross-validating for C for {} features".format(dataset))
        cross_validate_for_c(dataset, X[dataset])

        # print("Getting useful {} features.".format(dataset))
        # get_useful_features(dataset, X[dataset])
        #
        # print("Writing lists of useful {} features.".format(dataset))
        # for score in y_fields:
        #     print("     for " + score)
        #     list_useful_fields(score, dataset, X[dataset])

    # print("Writing final datasets for each score type.")
    # for score_type in y_fields:
    #     print("     for " + score_type)
    #     merge_final_datasets(score_type, list(X.keys())[0:-1])


y_fields = ["review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
            "review_scores_communication", "review_scores_location"]
feature_selection(y_fields)
