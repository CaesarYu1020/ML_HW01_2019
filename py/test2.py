#Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt
import tensorflow as tf

#Load Data
#["id","sale_yr","sale_month","sale_day","bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view","condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated","zipcode","lat","long","sqft_living15","sqft_lot15",'refurbish']
droptag=['id', 'sale_day','sale_yr','yr_renovated','refurbish','sale_month','yr_built']



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

import keras

model.add(Dense(516, input_dim=mydim, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(.2))
model.add(Dense(256, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(.2))
model.add(Dense(128, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(.2))
model.add(Dense(64, kernel_initializer='normal', activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(.2))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mae', optimizer=keras.optimizers.Adadelta(lr=30),)#keras.optimizers.Adadelta()
# # # 1. define the network

hist = model.fit(X_train, Y_train,batch_size=1000, epochs=1500 ,verbose=2,validation_data=(X_val,Y_val))
temp=model.predict(X_test)
print(model.evaluate(X_val,Y_val))
plt.figure('1')
plt.scatter(Y_train,model.predict(X_train),s=10,c='green')
plt.xlabel('price')
plt.ylabel('predice_price')
plt.plot([0,5*10**6],[0,5*10**6],c='orange')
plt.figure('2')
plt.scatter(Y_val,model.predict(X_val),s=10,c='green')
plt.xlabel('price')
plt.ylabel('predice_price')
plt.plot([0,5*10**6],[0,5*10**6],c='orange')
plt.figure('3')
plt.plot(hist.history['loss'],c='blue')
plt.plot(hist.history['val_loss'],c='red')
plt.show()

outputFlag=input('輸出請按 1 : ')
if outputFlag=='1':
  with open('Output.txt', 'w', newline='') as csvFile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvFile,delimiter='\n',quoting=csv.QUOTE_ALL)
    writer.writerow(['price'])
    writer.writerow(temp)

