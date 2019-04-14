#Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt
#Load Data
#["id","sale_yr","sale_month","sale_day","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15",'refurbish']
droptag=["id","sale_month","sale_day","yr_built","yr_renovated",'refurbish']

#['id', 'sale_day','sale_yr','yr_renovated','refurbish','sale_month','yr_built','view']
mydim=24-len(droptag)
df_train=pd.read_csv('./data/train-v3.csv')
df_train = df_train.drop(droptag, axis =1)
Y_train=df_train['price']
dataset=df_train.values
X_train = dataset[:,1:] #[4,5,6,7,8,9,10,11,12,13,14,18,19,20,21]
df_test=pd.read_csv('./data/test-v3.csv')
df_test = df_test.drop(droptag, axis =1)
dataset=df_test.values
X_test = dataset[:,:]
df_val=pd.read_csv('./data/valid-v3.csv')
df_val = df_val.drop(droptag, axis =1)
dataset=df_val.values
Y_val = df_val['price']
X_val = dataset[:,1:]
mscaler = preprocessing.MinMaxScaler().fit(X_train)
X_train=mscaler.transform(X_train)
X_val=mscaler.transform(X_val)
X_test=mscaler.transform(X_test)

#data標準化
# 
print(X_train)
print(Y_train)
# Model2

model = Sequential()
# model.add(Dense(output_dim=1, input_dim=13, init='uniform', activation='linear'))
# Compile model
# Input layer with dimension 1 and hidden layer i with 128 neurons.
# model.add(Dense(128, input_dim=mydim, activation='relu'))
# # Dropout of 20% of the neurons and activation layer.
# model.add(keras.layers.Dropout(.2))
# model.add(keras.layers.Activation("linear"))
# # Hidden layer j with 64 neurons plus activation layer.
# model.add(Dense(64, activation='relu'))
# model.add(keras.layers.Activation("linear"))
# # Hidden layer k with 64 neurons.
# model.add(Dense(64, activation='relu'))
# # Output Layer.
# model.add(Dense(1))

# model = Sequential()
# model.add(Dense(10, input_dim=mydim, activation='relu'))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(40, activation='relu'))
# model.add(Dense(1))
# Compile model

model = Sequential()
model.add(Dense(16, input_dim=mydim, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, kernel_initializer='normal', activation='relu'))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(Dense(64, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(32, kernel_initializer='normal', activation='linear'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mae', optimizer=keras.optimizers.Adadelta())#keras.optimizers.Adadelta()
# # # 1. define the network
hist = model.fit(X_train, Y_train,batch_size=32, epochs=150,verbose=2,validation_data=(X_val,Y_val))
temp=model.predict(X_test)
print(model.evaluate(X_val,Y_val))
with open('Output.txt', 'w', newline='') as csvFile:
  # 建立 CSV 檔寫入器
  writer = csv.writer(csvFile,delimiter='\n',quoting=csv.QUOTE_ALL)
  writer.writerow(['price'])
  writer.writerow(temp)

