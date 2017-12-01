import xgboost as xgb
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import argparse

import DataUtil
import Parameters
import PreprocessingUtil

np.random.seed(1)

def train_and_evaluate(trainFile, propertiesFile):
    """Train and evaluate a model.

        trainFile: path to train_2016_v2.csv
        propertiesFile: path to properties_2016.csv
        modelDir: directory to store model
    """
    x_train, x_test, y_train, y_test = _read_and_process_data(
        trainFile, propertiesFile)

    train = xgb.DMatrix(x_train, label=y_train)
    test = xgb.DMatrix(x_test, label=y_test)

    params = {}
    # eta is the 'learning rate in xgboost'
    params['eta'] = 0.02
    params['objective'] = 'reg:linear'
    params['eval_metric'] = 'mae'
    params['max_depth'] = 4
    params['silent'] = 1

    watchlist = [(train, 'train'), (test, 'test')]
    clf = xgb.train(params, train, 10000, watchlist, early_stopping_rounds=100,
        verbose_eval=10)
    print("test set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_test.values, y_pred =clf.predict(test))))
    print("training set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_train.values, y_pred =clf.predict(train))))

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
    standardScaler = StandardScaler()
    x_train.iloc[::] = standardScaler.fit_transform(x_train.iloc[::])
    x_test.iloc[::] = standardScaler.transform(x_test.iloc[::])

    y_train_inoutliners = PreprocessingUtil.applyIsolationForest(
        y_train.values.reshape(-1, 1))
    index = y_train_inoutliners == 1
    x_train = x_train.iloc[index]
    x_train.reset_index(drop = True, inplace = True)
    y_train = y_train.iloc[index]
    y_train.reset_index(drop = True, inplace = True)
    print("Finished reading and processing data")
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--trainFile',
        help='path to train_2016_v2.csv file', required=True)
    parser.add_argument('-pf', '--propertiesFile',
        help='path to properties_2016.csv.csv file', required=True)
    parser.add_argument()
    args = parser.parse_args()
    train_and_evaluate(args.trainFile, args.propertiesFile)
