import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.dummy import DummyRegressor


def setup(score_type):
    y_df = pd.read_csv("scores.csv")
    X = pd.read_csv("engineered_data/{}_dataset.csv".format(score_type))
    y = y_df[score_type]
    return train_test_split(X, y, test_size=0.2)


def cross_validate_for_c(X, y, score_type):
    # max_range = [1, 2]
    max_range = [1]
    for max_order in max_range:
        pf = PolynomialFeatures(max_order)
        Xpoly = pf.fit_transform(X)
        kf = KFold(n_splits=5)
        mean_error = []
        std_error = []

        # c_range = [0.1, 10, 100, 250, 500, 1000]
        c_range = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05]
        for C in c_range:
            print("               with max order = {} and C = {}.".format(max_order, C))
            model = Ridge(alpha=1 / (2 * C), max_iter=100000)
            fold = []

            for train, test in kf.split(Xpoly):
                model.fit(Xpoly[train], y.iloc[train])
                predictions = model.predict(Xpoly[test])
                fold.append(mean_squared_error(y.iloc[test], predictions))

            mean_error.append(np.array(fold).mean())
            std_error.append(np.array(fold).std())

        plt.errorbar(c_range, mean_error, yerr=std_error, linewidth=3,
                     label="max_order_{}_{}".format(max_order, score_type))

    plt.rc('font', size=12)
    plt.rcParams['figure.constrained_layout.use'] = True
    # plt.title("Cross validation for C for " + score_type)
    plt.title("Cross validation for Ridge Regression hyper parameters")
    plt.xlabel("C")
    plt.ylabel("Mean squared error")
    plt.legend(loc=1, fontsize='x-small')
    # plt.show()
    plt.savefig('ridge_regression/crossval_all_finetuned.png')
    # plt.savefig('ridge_regression/crossval_{}.png'.format(score_type))
    # plt.clf()


def ridge_regression(c, Xtrain, ytrain, Xtest, ytest):
    model = Ridge(alpha=1 / (2 * c), max_iter=100000)
    model.fit(Xtrain, ytrain)
    return [mean_squared_error(ytrain, model.predict(Xtrain)), mean_squared_error(ytest, model.predict(Xtest))]


def neural_net(epochs, Xtrain, ytrain, Xtest, ytest):
    model = Sequential()
    model.add(Dense(8, activation='relu', input_shape=Xtrain.shape[1:]))
    model.add(Dense(4, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='sgd', loss='mean_squared_error')
    history = model.fit(Xtrain, ytrain, epochs=epochs, validation_data=(Xtest, ytest))
    return model, history


def cross_validate_epochs(epochs_total, history, score):
    epochs = np.arange(2, epochs_total)
    mse_train = history.history['loss']
    mse_test = history.history['val_loss']

    # plt.plot(epochs, mse_train[2:], label="training")
    plt.plot(epochs, mse_test[2:], label=score)
    plt.title("Model performance on test datasets")
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(fontsize='x-small')
    plt.savefig("neural_net/all.png")
    # plt.clf()


def baseline(strategy, Xtrain, ytrain, Xtest, ytest):
    model = DummyRegressor(strategy=strategy)
    model.fit(Xtrain, ytrain)
    return [mean_squared_error(ytrain, model.predict(Xtrain)), mean_squared_error(ytest, model.predict(Xtest))]


def generate_bar_chart(ridge_training, nn_training, mean_training, median_training,
                       ridge_test, nn_test, mean_test, median_test):
    labels = ['Rating', 'Accuracy', 'Cleanliness', 'Check in', 'Comm', 'Location']
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots()
    ax.bar(x - 1.5*width, ridge_training, width, label='Ridge regression - training')
    ax.bar(x - 0.5*width, nn_training, width, label='Neural net - training')
    ax.bar(x + 0.5*width, mean_training, width, label='Mean - training')
    ax.bar(x + 1.5*width, median_training, width, label='Median - training')
    ax.bar(x - 1.5*width, ridge_test, width, label='Ridge regression - test')
    ax.bar(x - 0.5*width, nn_test, width, label='Neural net - test')
    ax.bar(x + 0.5*width, mean_test, width, label='Mean - test')
    ax.bar(x + 1.5*width, median_test, width, label='Median - test')
    ax.set_title('MSE of each model')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('MSE')
    ax.legend()
    plt.savefig("results.png")
    # plt.show()


def run_models(y_fields):
    # epochs = [1000, 1000, 1000, 1000, 1000, 1000]
    epochs = [100, 100, 100, 100, 100, 100]
    # epochs = [10, 10, 10, 10, 10, 10]
    # epochs = [1500, 3500, 500, 500, 2000, 500]

    ridge_train = []
    ridge_test = []
    nn_train = []
    nn_test = []
    mean_train = []
    mean_test = []
    median_train = []
    median_test = []

    for index, score_type in enumerate(y_fields):
        print("     for " + score_type)
        X_train, X_test, y_train, y_test = setup(score_type)

        # print("          cross-validating for c")
        # cross_validate_for_c(X_train, y_train, score_type)

        print("          training and testing ridge regression model")
        selected_c = [0.01, 0.001, 0.001, 0.01, 0.01, 0.001]
        mse = ridge_regression(selected_c[index], X_train, y_train, X_test, y_test)
        ridge_train.append(mse[0])
        ridge_test.append(mse[1])

        print("          training and testing neural net model")
        model, history = neural_net(epochs[index], X_train, y_train, X_test, y_test)
        # cross_validate_epochs(epochs[index], history, score_type)
        nn_train.append(mean_squared_error(y_train, model.predict(X_train)))
        nn_test.append(mean_squared_error(y_test, model.predict(X_test)))

        print("          running baselines")
        mse = baseline('mean', X_train, y_train, X_test, y_test)
        mean_train.append(mse[0])
        mean_test.append(mse[1])
        mse = baseline('median', X_train, y_train, X_test, y_test)
        median_train.append(mse[0])
        median_test.append(mse[1])

    print("          generating bar chart")
    generate_bar_chart(ridge_train, nn_train, mean_train, median_train,
                       ridge_test, nn_test, mean_test, median_test)
