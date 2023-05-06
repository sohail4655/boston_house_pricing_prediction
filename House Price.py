import warnings

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
housing = pd.read_csv("data.csv")
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing = strat_train_set.copy()
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = my_pipeline.fit_transform(housing)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
def print_scores(scores):
    print("RMSE Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std(), "\n")
print_scores(rmse_scores)
X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Predicted Prices on Test dataset:")
print(final_predictions, "\n")
print("Original Prices:")
print(list(Y_test),"\n")
print("The RMSE of the model on Test dataset:")
print(final_rmse, "\n")
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
print("Predicted Price is: ", model.predict(features))
print("Original Price is: ", list(some_labels[:1]), "\n")

import tensorflow as tf
from sklearn import preprocessing
housing = preprocessing.normalize(housing)
X_test = preprocessing.normalize(X_test)
ann_data = housing[:5]
ann_labels = housing_labels[:5]
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
def HousePricePredictionModel():
    model = Sequential()
    model.add(Dense(128,activation='relu',input_shape=(housing[0].shape)))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
k = 4
num_val_samples = len(housing)
num_epochs = 50
all_scores = []
model = HousePricePredictionModel()
history = model.fit(x=housing, y=housing_labels, epochs=num_epochs, batch_size=1, verbose=1, validation_data=(X_test,Y_test))
housing_predictions = model.predict(housing)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)
print("\nRMSE of the FF Neural network model: ", rmse, "\n")
final_predictions = model.predict(X_test)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Predicted Prices on Test dataset:")
print(final_predictions, "\n")
print("Original Prices:")
print(list(Y_test),"\n")
print("The RMSE of the FF Neural network model on Test dataset:")
print(final_rmse, "\n")
features = np.array([[9.42440546e-05, 1.56454127e-01, 7.11866277e-03, 0.00000000e+00,
       7.66625221e-04, 1.19452726e-02, 6.25816507e-02, 1.80319248e-02,
       1.95567658e-03, 6.16038124e-01, 3.20730960e-02, 7.68365773e-01,
       1.28487952e-02]])
print("Predicted Price is: ", model.predict(features))
print("Original Price is: ", list(ann_labels[:1]), "\n")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
housing_log = pd.read_csv("data_log.csv")
X = housing_log.iloc[:, :-1]
y = housing_log.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print("\nClassification report:\n", classification_report(y_test, predictions), "\n")
print("Accuracy of the Model: ", accuracy_score(y_test, predictions), "\n")
print("The predictions for the Test dataset are: ", predictions, "\n")