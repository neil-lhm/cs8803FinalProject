"""Util functions for reading data into pandas dataframes.
"""

import pandas as pd

from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

def getTrainAndTestData(x, y, testSetRatio):
    """Splits data for training and testing.

        testSetRatio: float, percentage of data reserved for test set
    """
    return train_test_split(x, y, test_size=testSetRatio,
        random_state=RANDOM_STATE)

def readTrainFile(trainFile):
    """Read the train data from the file and return a pandas DataFrame.

        trainFile: string, path to the train file
    """

    df_train = pd.read_csv(trainFile, header = 0, skipinitialspace=True,
        engine="python", parse_dates=["transactiondate"])
    return df_train

def readPropertiesFile(propertiesFile):
    """Read the properties data from the file and return a pandas DataFrame.

        propertiesFile: string, path to the properties file
    """
    columnDtypes = {
        'parcelid':int, 'airconditioningtypeid':str,
        'architecturalstyletypeid':str, 'basementsqft':float,
        'bathroomcnt':float, 'bedroomcnt':float, 'buildingclasstypeid':str,
        'buildingqualitytypeid':str, 'calculatedbathnbr':float,
        'decktypeid':str, 'finishedfloor1squarefeet':float,
        'calculatedfinishedsquarefeet':float, 'finishedsquarefeet12':float,
        'finishedsquarefeet13':float, 'finishedsquarefeet15':float,
        'finishedsquarefeet50':float, 'finishedsquarefeet6':float, 'fips':str,
        'fireplacecnt':float, 'fullbathcnt':float, 'garagecarcnt':float,
        'garagetotalsqft':float, 'hashottuborspa':str,
        'heatingorsystemtypeid':str, 'latitude':float, 'longitude':float,
        'lotsizesquarefeet':float, 'poolcnt':float,
        'poolsizesum':float, 'pooltypeid10':str, 'pooltypeid2':str,
        'pooltypeid7':str, 'propertycountylandusecode':str,
        'propertylandusetypeid':str, 'propertyzoningdesc':str,
        'rawcensustractandblock':float, 'regionidcity':str,
        'regionidcounty':str, 'regionidneighborhood':str, 'regionidzip':str,
        'roomcnt':float, 'storytypeid':float, 'threequarterbathnbr': float,
        'typeconstructiontypeid':float, 'unitcnt':float,
        'yardbuildingsqft17':float, 'yardbuildingsqft26':float,
        'yearbuilt':float, 'numberofstories': float, 'fireplaceflag':str,
        'structuretaxvaluedollarcnt':float, 'taxvaluedollarcnt': float,
        'assessmentyear':float, 'landtaxvaluedollarcnt': float,
        'taxamount':float, 'taxdelinquencyflag':str,
        'taxdelinquencyyear': float, 'censustractandblock':float}
    # use c engine because datatypes are defined
    print("Reading properties file.")
    df_properties = pd.read_csv(propertiesFile, header = 0,
        skipinitialspace=True, dtype=columnDtypes, engine="c")
    print("Finished reading properties file.")

    return df_properties
