import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

import argparse
import tempfile

import DataUtil
import Parameters
import PreprocessingUtil

np.random.seed(1)
tf.set_random_seed(1)

def train_and_evaluate(trainFile, propertiesFile, model_dir):
    """Train and evaluate a model.

        trainFile: path to train_2016_v2.csv
        propertiesFile: path to properties_2016.csv
        modelDir: directory to store model
    """
    x_train, x_test, y_train, y_test = _read_and_process_data(
        trainFile, propertiesFile)

    x = tf.placeholder(tf.float32, [None, 58])
    y = tf.placeholder(tf.float32, [None, 1])
    y_pred = _get_dnn(x)

    dataset = tf.contrib.data.Dataset.from_tensor_slices((x, y))
    batch_size = 50
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    with tf.name_scope('loss'):
        mse = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    mse = tf.reduce_mean(mse)

    with tf.name_scope('mae'):
        mae = tf.metrics.mean_absolute_error(labels=y, predictions=y_pred)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(0.1).minimize(mse)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={x: x_train, y: y_train})
        for i in range(1600):
            batch = iterator.get_next()
            if i % 100 == 0:
                train_loss = mse.eval(feed_dict={
                    x: batch[0].eval(), y: batch[1].eval()
                })
                print("batch {}, mean squared error: {}".format(i, train_loss))
            train_step.run(feed_dict={x: batch[0].eval(), y: batch[1].eval()})
        y_test_pred = sess.run(y_pred, feed_dict={x:x_test, y:y_test})
        y_train_pred = sess.run(y_pred, feed_dict={x:x_train, y:y_train})
    print("test set error: {}".format(
        mean_absolute_error(y_true = y_test, y_pred =y_test_pred)))
    print("train set error: {}".format(
        mean_absolute_error(y_true = y_train, y_pred =y_train_pred)))


def _get_dnn(x):
    """Return a deep neural network
    """

    dense1 = tf.layers.dense(
        x, 1024, activation = tf.tanh,
        kernel_initializer = tf.random_normal_initializer()
    )
    batch1 = tf.layers.batch_normalization(dense1)
    dropout1 = tf.layers.dropout(batch1, rate = 0.1)

    dense2 = tf.layers.dense(
        dropout1, 512, activation = tf.nn.relu,
        kernel_initializer = tf.random_normal_initializer()
    )
    batch2 = tf.layers.batch_normalization(dense2)
    dropout2 = tf.layers.dropout(batch2, rate = 0.3)


    dense3 = tf.layers.dense(
        dropout2, 256, activation = tf.nn.relu,
        kernel_initializer = tf.random_normal_initializer()
    )
    batch3 = tf.layers.batch_normalization(dense3)
    dropout3 = tf.layers.dropout(batch3, rate = 0.2)

    dense4 = tf.layers.dense(
        dropout3, 128, activation = tf.nn.relu,
        kernel_initializer = tf.random_normal_initializer()
    )
    batch4 = tf.layers.batch_normalization(dense4)
    dropout4 = tf.layers.dropout(batch4, rate = 0.1)

    y_pred = tf.layers.dense(dropout4, 1)

    return y_pred

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

    x_train = x_train.values
    x_test = x_test.values
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', '--trainFile',
        help='path to train_2016_v2.csv file', required=True)
    parser.add_argument('-pf', '--propertiesFile',
        help='path to properties_2016.csv.csv file', required=True)
    parser.add_argument("--modelDir", help="dir to store trained model.\
        Default to a temporary directory", default='')

    args = parser.parse_args()
    train_and_evaluate(args.trainFile, args.propertiesFile, args.modelDir)
