# cs8803FinalProject

## Data
Both train_2016_v2.csv and properties_2016.csv can be downloaded from 
https://www.kaggle.com/c/zillow-prize-1/data. Download and unzip 
train_2016_v2.csv.zip and properties_2016.csv.zip.

## Util Scripts
DataUtil.py, PreprocessingUtil.py and Parameters.py are util scripts. 
They are not executable.

## Model Implementations
DNNBatchNormalization.py, DNNEmbeddings.py, DNNLinearXgboost.py, SvmKnnRr.py
DNNLinearCombined.py and Xgboost.py are executable files where different models 
are implemented. They all take in two required command-line arguments: -tf 
and -pf. Pass train_2016_v2.csv with '-tf train_2016_v2.csv' and 
properties_2016.csv with '-pf' properties_2016.csv. 

An example default run for DNNLinearCombined is:
"python DNNLinearCombined.py -tf path_totrain_2016_v2.csv -pf path_to properties_2016.csv".

#### DNNBatchNormalization.py
This script builds a DNN model with batch normalization technique. Data preprocessing:
StandardScaler and LabelEncoder. 

#### DNNEmbeddings.py
This script can build 3 types models: "wide"(Linear Regression), "deep"(Deep Neural Network)
and "combined" (Jointly trained DNNLinear ensemble model). Categorical features are mapped 
to low-dimensional dense vector to be fed into DNNs(embeddings). Linear Regression Model 
only trains on the numeric featrues. Data preprocessing: MinMaxScaler and Isolation Forest.
Read the documentation for more details.

#### DNNLinearXgboost.py
This script builds a model where results from a DNNLinear ensemble are used
as a meta-feature in xgboost training. Data preprocessing: LabelEncoder, MinMaxScaler 
and Isolation Forest. Read the documentation for more details.

#### SvmKnnRr.py
This script explores the power of three traditional machine learning models:
SVM, KNN and Random Forest. Data preprocessing: LabelEncoder, MinMaxScaler.

#### DNNLinearCombined.py
This script can build 3 types models: "wide"(Linear Regression), "deep"(Deep Neural Network)
and "combined" (Jointly trained DNNLinear ensemble model). Categorical features are mapped 
to integers by LabelEncoder. Data preprocessing: MinMaxScaler and Isolation Forest.
Read the documentation for more details.

#### Xgboost.py
Experiment on Xgboost: http://xgboost.readthedocs.io/en/latest/model.html. 
Data preprocessing: StandardScaler, LabelEncoder and Isolation Forest.
