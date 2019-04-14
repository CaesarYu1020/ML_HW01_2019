#Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import normalize
#Load Data
# df_train=pd.read_csv('./data/train-v3.csv')
# df_train = df_train.drop(['id','sale_day','sale_yr','sale_month','long','lat','yr_built','yr_renovated','zipcode'], axis =1)
# Y_train=df_train['price']
# dataset=df_train.values
# X_train = dataset[:,1:]#[4,5,6,7,8,9,10,11,12,13,14,18,19,20,21]
# df_train.head()
# print(X_train)
X = [[ 1., -1., 2.],
     [ 2. , 0., 0.],
     [ 0.,   1., -1. ]]
Y=[[1,1,1],[1,1,1],[1,1,1]]
scaler=preprocessing.MinMaxScaler().fit(X);
scaler2=preprocessing.MinMaxScaler().fit(Y);

print(scaler.scale_)
print(scaler2.scale_)
print(scaler.transform(Y))
print(scaler2.transform(Y))