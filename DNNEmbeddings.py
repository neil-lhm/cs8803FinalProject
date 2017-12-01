"""This script supports three models: linear regression, deep neural network
and DNNLinearCombined. Categorical features are handled by embeddings. Thus the
linear regression model is trained only on numeric section of the data.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

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
    inter, y, numeric_features, deep_columns = \
        _read_and_process_data(trainFile, propertiesFile)
    x_train, x_test, y_train, y_test = DataUtil.getTrainAndTestData(
        inter, y, Parameters.getTestSetRatio()
    )
    numeric_cols = x_train.select_dtypes(exclude=['object'])
    # normalization with standardScaler.
    standardScaler = StandardScaler()
    numeric_cols.iloc[::] = standardScaler.fit_transform(numeric_cols.iloc[::])
    x_train[numeric_cols.columns] = numeric_cols

    numeric_cols = x_test.select_dtypes(exclude=['object'])
    numeric_cols.iloc[::] = standardScaler.transform(numeric_cols.iloc[::])
    x_test[numeric_cols.columns] = numeric_cols

    if model_dir:
        print("Storing model to: {}".format(model_dir))
    else:
        model_dir = tempfile.mkdtemp()
        print("Storing model to temporary directory: {}".format(model_dir))
    print("Training started")
    m = _build_estimator(model_dir, model_type, numeric_features, deep_columns)
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



def _build_estimator(model_dir, model_type, numeric_features, deep_columns):
    hidden_units = [360, 180, 90, 45]

    if model_type == 'wide':
        return tf.estimator.LinearRegressor(
                model_dir=model_dir,
                feature_columns=numeric_features)
    elif model_type == 'deep':
        return tf.estimator.DNNRegressor(
                model_dir=model_dir,
                feature_columns=deep_columns,
                hidden_units=hidden_units,
                dropout=0.3)
    else:
        return tf.estimator.DNNLinearCombinedRegressor(
                model_dir=model_dir,
                linear_feature_columns=numeric_features,
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=hidden_units,
                dnn_dropout=0.3)

def _input_fn(x, y, num_epochs, shuffle=True, batch_size=32, num_threads=1):
    return tf.estimator.inputs.pandas_input_fn(
        x=x,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=num_threads)

def _read_and_process_data(trainFile, propertiesFile):
    """Read and process data. Map categorical features to embeddings and
    indicators for dnn.

    Return:
        inter: pandas.DataFrame, "x" to be split in train_test_split
        y: pandas.Series, "y" to be split in train_test_split
        numeric_features: features for linear regression to train on
        deep_columns: features for deep neural network tro train on
    """

    print("Start reading and processing data")
    df_train = DataUtil.readTrainFile(trainFile)
    df_properties = DataUtil.readPropertiesFile(propertiesFile)

    inter = pd.merge(df_properties, df_train, how="inner", on=["parcelid"]);
    inter['transactiondate'] = pd.to_datetime(df_train["transactiondate"])
    inter['transaction_year'] = inter['transactiondate'].dt.year
    inter['transaction_month'] = inter['transactiondate'].dt.month
    inter['transaction_day'] = inter['transactiondate'].dt.day

    y = inter['logerror']
    inter = inter.drop(Parameters.getColumnsToDrop(), axis=1)

    numeric_cols = inter.select_dtypes(exclude=['object'])
    numeric_cols = numeric_cols.fillna(-1)
    inter[numeric_cols.columns] = numeric_cols

    numeric_features = [
        tf.feature_column.numeric_column(col) for col in numeric_cols.columns
    ]

    categorical_cols = inter.select_dtypes(include=['object'])
    categorical_cols = categorical_cols.fillna('none')
    inter[categorical_cols.columns] = categorical_cols

    complex_features = ["regionidcity", "regionidneighborhood", "regionidzip"]
    simple_categorical_features = [
        tf.feature_column.categorical_column_with_hash_bucket(col,
            hash_bucket_size=100)
            for col in categorical_cols if col not in complex_features
    ]
    complex_categorical_features = [
        tf.feature_column.categorical_column_with_hash_bucket(col,
            hash_bucket_size=500) for col in complex_features
    ]

    deep_indicator_columns = [
        tf.feature_column.indicator_column(col)
        for col in simple_categorical_features
    ]

    deep_embedding_columns = [
        tf.feature_column.embedding_column(col, dimension=8)
        for col in complex_categorical_features
    ]

    deep_columns = numeric_features + deep_indicator_columns +\
        deep_embedding_columns

    return inter, y, numeric_features, deep_columns

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
