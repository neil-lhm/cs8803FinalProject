"""Util functions for applying preprocessing techniques: MinMaxScaler,
StandardScaler, Isolation Forest and LabelEncoder.
"""

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest

def fillNAsWithNegativeOnes(dataFrame):
    """Fill nans with -1s in the input DataFrame and return the DataFrame.

        dataFrame: pandas.DataFrame with only numeric columns.
    """
    dataFrame = dataFrame.fillna(-1)
    return dataFrame

def fillNAsWithNones(dataFrame):
    """Fill nans with "None"s in the input DataFrame and return the DataFrame

        dataFrame: pandas.DataFrame with only object columns.
    """
    dataFrame = dataFrame.fillna("None")
    return dataFrame

def applyLabelEncoder(df_properties):
    """Apply LabelEncoder to properties DataFrame and
    return the modified DataFrame.

        df_properties: pandas.DataFrame, the properties DataFrame to apply
            LableEncoder on
    """
    for column in df_properties.columns:
        df_properties[column] = df_properties[column].fillna(-1)
        if 'object' == df_properties[column].dtype:
            labelEncoder = LabelEncoder()
            target = list(df_properties[column].values)
            labelEncoder.fit(target)
            df_properties[column] = labelEncoder.transform(target)
    return df_properties

def applyStandardScaler(array):
    """Apply StandardScaler on the input numpy array and return the modified
    array.
    """
    standardScaler = StandardScaler()
    array = standardScaler.fit_transform(array)
    return array

def applyMinMaxScaler(array):
    """Apply MinMaxScaler on the input numpy array and return the modified
    array.
    """
    minMaxScaler = MinMaxScaler()
    array = minMaxScaler.fit_transform(array)
    return array

def applyIsolationForest(array):
    """Apply IsolationForest algorithm on the input numpy array and return
    the modified array
    """
    clf = IsolationForest(max_samples = 1024, random_state = 2)
    clf.fit(array)
    # an array where isInliners = 1 if train[i] is inliner and -1 otherwise
    isInliners = clf.predict(array)
    return isInliners
