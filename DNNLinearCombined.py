"""This script supports three models: linear regression, deep neural network
and DNNLinearCombined. Categorical features are handled by LabelEncoders. Thus
all models are trained on the full set of features.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import argparse
import tempfile
import time

import DataUtil
import Parameters
import PreprocessingUtil

np.random.seed(1)
tf.set_random_seed(1)

def train_and_evaluate(trainFile, propertiesFile, model_type, model_dir):
    """Train and evaluate a model.

        trainFile: path to train_2016_v2.csv
        propertiesFile: path to properties_2016.csv
        modelType: 'linear', 'deep', or 'combined'
        modelDir: directory to store model
    """
    x_train, x_test, y_train, y_test = _read_and_process_data(
        trainFile, propertiesFile
    )
    if model_dir:
        print("Storing model to: {}".format(model_dir))
    else:
        model_dir = tempfile.mkdtemp()
        print("Storing model to temporary directory: {}".format(model_dir))
    print("Training started")
    m = _build_estimator(model_dir, model_type, x_train)
    start_time = time.time()
    m.train(input_fn=_input_fn(x_train, y_train, num_epochs=5))
    print("Training takes {} seconds".format(time.time() - start_time))
    print("Training done. Starting Evaluation")
    # evaluation
    y_test_pred = list(m.predict(input_fn=_input_fn(
        x_test, y_test, shuffle = False, num_threads=1, num_epochs = 1)))
    y_test_pred = np.array([x["predictions"][0] for x in y_test_pred])
    mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
    print("test set mean absolute error is {}".format(mae))
    # pring train set error as well
    y_train_pred = list(m.predict(input_fn=_input_fn(
        x_train, y_train, shuffle = False, num_threads=1, num_epochs = 1)))
    y_train_pred = np.array([x["predictions"][0] for x in y_train_pred])
    mae = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
    print("train set mean absolute error is {}".format(mae))
    return y_train_pred

def _build_estimator(model_dir, model_type, x_train):
    """Build and return a model to train on.
    """
    numeric_columns = _build_numeric_columns(x_train)
    hidden_units = [1024, 512, 256, 128]

    if model_type == 'wide':
        return tf.estimator.LinearRegressor(
                model_dir=model_dir,
                feature_columns=numeric_columns)
    elif model_type == 'deep':
        return tf.estimator.DNNRegressor(
                model_dir=model_dir,
                feature_columns=numeric_columns,
                hidden_units=hidden_units,
                dropout=0.5)
    else:
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
    numeric_columns = [
        tf.feature_column.numeric_column(col) for col in x_train.columns
    ]
    return numeric_columns

def _input_fn(x, y, num_epochs, shuffle=True, batch_size=32, num_threads=1):
    return tf.estimator.inputs.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=num_threads)

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
    print("MinMaxScaler done. Start filtering outliers")
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
    parser.add_argument("-m", "--modelTpye",
        help="Options are [wide, deep, combined].\
        'wide' selects the linear regression model, \
        'deep' selects the deep neural network model \
        and 'combined' selects the ensemble model(dnn & linear). Default\
        to 'combined'.", default='combined')
    parser.add_argument("--modelDir", help="dir to store trained model.\
        Default to a temporary directory", default='')

    args = parser.parse_args()
    train_and_evaluate(args.trainFile, args.propertiesFile,
        args.modelTpye, args.modelDir)
