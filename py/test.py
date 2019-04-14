#Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import keras
import pandas as pd
from sklearn import preprocessing
import csv
import matplotlib.pyplot as plt
#Load Data
droptag=['id', 'sale_day', 'sale_month','sale_yr','yr_renovated']
mydim=22-len(droptag)
df_train=pd.read_csv('./data/train-v3.csv')
df_train = df_train.drop(droptag, axis =1)
Y_train=df_train['price']
dataset=df_train.values
X_train = dataset[:,1:] #[4,5,6,7,8,9,10,11,12,13,14,18,19,20,21]
df_test=pd.read_csv('./data/test-v3.csv')
df_test = df_test.drop(droptag, axis =1)
#id,sale_yr,sale_month,sale_day,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,
# grade,sqft_above,sqft_basement,yr_built,yr_renovated,zipcode,lat,long,sqft_living15,sqft_lot15
dataset=df_test.values
X_test = dataset[:,:]
df_val=pd.read_csv('./data/valid-v3.csv')
df_val = df_val.drop(droptag, axis =1)
dataset=df_val.values
Y_val = df_val['price']
X_val = dataset[:,1:]
min_max_scaler = preprocessing.MinMaxScaler().fit(X_train)
X_train=min_max_scaler.transform(X_train)
X_val=min_max_scaler.transform(X_val)
X_test=min_max_scaler.transform(X_test)

#data標準化
# 
print(X_train)
print(Y_train)
# Model2

model = Sequential()
# model.add(Dense(output_dim=1, input_dim=13, init='uniform', activation='linear'))
# Compile model
model.add(Dense(200, input_dim=mydim, kernel_initializer='normal', activation='relu'))
model.add(Dense(150, kernel_initializer='normal', activation='relu'))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dense(150, kernel_initializer='normal', activation='sigmoid'))
model.add(Dense(50, kernel_initializer='normal', activation='linear'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mae', optimizer=keras.optimizers.Adadelta())#keras.optimizers.Adadelta()
# # # 1. define the network
hist = model.fit(X_train, Y_train,batch_size=1000, epochs=330,verbose=2,validation_data=(X_val,Y_val))
temp=model.predict(X_test)
print(model.evaluate(X_val,Y_val))
with open('Output.txt', 'w', newline='') as csvFile:
  # 建立 CSV 檔寫入器
  writer = csv.writer(csvFile,delimiter='\n',quoting=csv.QUOTE_ALL)
  writer.writerow(['price'])
  writer.writerow(temp)

