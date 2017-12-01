"""Ensemble model built with DNNLinearCombined model and Xgboost model,
where training result from the DNNLinearCombined is used as a meta-feature
in xgboost training.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import argparse
import tempfile
import time

import DataUtil
import Parameters
import PreprocessingUtil

import DNNLinearCombined

np.random.seed(1)
tf.set_random_seed(1)

def train_and_evaluate(trainFile, propertiesFile):
    """Train and evaluate a model.

        trainFile: path to train_2016_v2.csv
        propertiesFile: path to properties_2016.csv
    """
    x_train, x_test, y_train, y_test = _read_and_process_data(
        trainFile, propertiesFile
    )
    m = _build_DNNLinear_estimator(x_train)
    start_time = time.time()
    m.train(input_fn=_input_fn(x_train, y_train, num_epochs=5))
    dnn_linear_train_time = time.time() - start_time
    print("DNNLinear model training takes {} seconds".format(
        dnn_linear_train_time))
    # use result from DNNLinear model as the meta-feature
    y_train_pred = list(m.predict(input_fn=_input_fn(
        x_train, y_train, shuffle = False, num_threads=1, num_epochs = 1)))
    y_train_pred = np.array([x["predictions"][0] for x in y_train_pred])
    meta_feature = "DNNLinear"
    x_train[meta_feature] = y_train_pred
    y_test_pred = list(m.predict(input_fn=_input_fn(
        x_test, y_test, shuffle = False, num_threads=1, num_epochs = 1)))
    y_test_pred = np.array([x["predictions"][0] for x in y_test_pred])
    x_test[meta_feature] = y_test_pred
    # prepare xgb training
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
    start_time = time.time()
    clf = xgb.train(params, train, 10000, watchlist, early_stopping_rounds=100,
        verbose_eval=10)
    xgboost_train_time = time.time() - start_time
    print("Xgboost training takes {} seconds".format(xgboost_train_time))
    print("DNNLinear and Xgboost ensemble takes {} seconds in total".format(
        dnn_linear_train_time + xgboost_train_time
    ))
    print("test set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_test.values, y_pred =clf.predict(test))))
    print("training set mean_abosulte_error is: {}".format(
        mean_absolute_error(y_true = y_train.values, y_pred =clf.predict(train))))

def _build_DNNLinear_estimator(x_train):
    """Build and return a DNNLinear model to train on.
    """
    model_dir = tempfile.mkdtemp()
    numeric_columns = _build_numeric_columns(x_train)
    hidden_units = [1024, 512, 256, 128]
    return tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            linear_feature_columns=numeric_columns,
            dnn_feature_columns=numeric_columns,
            dnn_hidden_units=hidden_units,
            dnn_dropout=0.5)

def _build_numeric_columns(x_train):
    """Build and return feature columns given x_train. In this experiment
    all columns are numeric because LabelEncoder is used.
    """
    return DNNLinearCombined._build_numeric_columns(x_train)

def _input_fn(x, y, num_epochs, shuffle=True, batch_size=32, num_threads=1):
    return DNNLinearCombined._input_fn(
            x, y, num_epochs, shuffle, batch_size, num_threads)

def _read_and_process_data(trainFile, propertiesFile):
    """Read and preprocess data with LabelEncoder, MinMaxScaler
        and IsolationForest. Return x_train, x_test, y_train and y_test.

        Return:
            x_train, x_test, y_train, y_test (80/20 split)
    """
    return DNNLinearCombined._read_and_process_data(trainFile, propertiesFile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--trainFile',
        help='path to train_2016_v2.csv file', required=True)
    parser.add_argument('-pf', '--propertiesFile',
        help='path to properties_2016.csv.csv file', required=True)

    args = parser.parse_args()
    train_and_evaluate(args.trainFile, args.propertiesFile)
