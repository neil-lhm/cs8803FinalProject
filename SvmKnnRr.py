"""Experiments on classis machine models: SVM, KNN and Random Forests.
"""
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler

import argparse

import DataUtil
import Parameters
import PreprocessingUtil


from sklearn.metrics import mean_absolute_error

def train_and_evaluate(trainFile, propertiesFile):
    x_train, x_test, y_train, y_test = _read_and_process_data(
        trainFile, propertiesFile)
    # RandomForestRegressor
    print("Random Forest:")
    for i in range(1, 16):
        neigh = KNeighborsRegressor(n_neighbors=i)
        neigh.fit(x_train.values, y_train.values)
        print("test set mean_abosulte_error is: {}".format(mean_absolute_error(
            y_true = y_test.values, y_pred = neigh.predict(x_test.values))))
        print("training set mean_abosulte_error is: {}".format(
            mean_absolute_error(
                y_true = y_train.values, y_pred = neigh.predict(x_train.values))))

    # KNeighborsRegressor
    print("KNN:")
    for i in range(1, 16):
        neigh = KNeighborsRegressor(n_neighbors=i)
        neigh.fit(x_train.values, y_train.values)
        print("test set mean_abosulte_error is: {}".format(mean_absolute_error(
                y_true = y_test.values, y_pred = neigh.predict(x_test.values))))
        print("training set mean_abosulte_error is: {}".format(
            mean_absolute_error(
                y_true = y_train.values, y_pred = neigh.predict(x_train.values))))

    # SVR
    print("SVR_rbf")
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(x_train.values, y_train.values)
    y_rbf = svr_rbf.predict(x_test.values)
    print("test set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_test, y_pred = y_rbf)))
    print("training set mean_abosulte_error is: {}".format(
        mean_absolute_error(
            y_true = y_train, y_pred = svr_rbf.predict(x_train.values))))

    print("SVR_lin")
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_lin.fit(x_train.values, y_train.values)
    y_lin = svr_lin.predict(x_test.values)
    print("test set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_test, y_pred = y_lin)))
    print("training set mean_abosulte_error is: {}".format(
        mean_absolute_error(
            y_true = y_train, y_pred = svr_lin.predict(x_train.values))))

    print("SVR_poly")
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_poly.fit(x_train.values, y_train.values)
    y_poly = svr_polypredict(x_test.values)
    print("test set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_test, y_pred = y_poly)))
    print("training set mean_abosulte_error is: {}".format(
        mean_absolute_error(
            y_true = y_train, y_pred = svr_poly.predict(x_train.values))))


def _read_and_process_data(trainFile, propertiesFile):
    """Read and preprocess data with LabelEncoder, MinMaxScaler
        and IsolationForest. Return x_train, x_test, y_train and y_test.
    """
    print("Start reading and processing data")
    df_train = DataUtil.readTrainFile(trainFile)
    df_properties = DataUtil.readPropertiesFile(propertiesFile)

    df_properties = PreprocessingUtil.applyLabelEncoder(df_properties)

    # find the intersection of training data and property information for
    # all data
    inter = pd.merge(df_properties, df_train, how="inner", on=["parcelid"])
    # decompose transaction date information
    inter['transactiondate'] = pd.to_datetime(df_train["transactiondate"])
    inter['transaction_year'] = inter['transactiondate'].dt.year
    inter['transaction_month'] = inter['transactiondate'].dt.month
    inter['transaction_day'] = inter['transactiondate'].dt.day

    y = inter['logerror'].astype(float)
    x_train, x_test, y_train, y_test = DataUtil.getTrainAndTestData(
        inter, y, Parameters.getTestSetRatio()
    )
    x_train = x_train.drop(Parameters.getColumnsToDrop(), axis = 1)
    x_test = x_test.drop(Parameters.getColumnsToDrop(), axis = 1)
    print("Start applying MinMaxScaler")
    minMaxScaler = MinMaxScaler()
    x_train.iloc[::] = minMaxScaler.fit_transform(x_train.iloc[::])
    x_test.iloc[::] = minMaxScaler.transform(x_test.iloc[::])

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--trainFile',
        help='path to train_2016_v2.csv file', required=True)
    parser.add_argument('-pf', '--propertiesFile',
        help='path to properties_2016.csv.csv file', required=True)
    args = parser.parse_args()
    train_and_evaluate(args.trainFile, args.propertiesFile)
