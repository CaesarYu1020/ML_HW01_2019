import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing as preprocessing
import numpy as np
droptag=["id","sale_month","sale_day","yr_built","yr_renovated",'refurbish']
plt.close('all')
#['id', 'sale_day','sale_yr','yr_renovated','refurbish','sale_month','yr_built','view']
mydim=24-len(droptag)
df_train=pd.read_csv('./data/train-v3.csv')
df_train = df_train.drop(droptag, axis =1)
Y_train=df_train['price']
dataset=df_train.values
df_train.keys()
X_train = dataset[:,1:]
y=Y_train

plt.figure('各特徵與價錢關係')
t=1;
for tag in df_train.keys():
   x=df_train.get(tag)
   plt.subplot(5,4,t)
   x=np.array(x).reshape(len(x),1)
   mscaler = preprocessing.MinMaxScaler().fit(x)
   x = mscaler.transform(x)
   t=t+1;
   plt.scatter(x, y,s=2)
   plt.plot([0,1],[0.5*10**7,0.5*10**7],c='r')
   plt.xlabel(tag)
   plt.ylabel('price')
plt.subplots_adjust(left=0.11, bottom=0.06, right=0.94, top=0.91, wspace=0.59, hspace=0.75)
data = df_train
correlations = data.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1,cmap='GnBu')
fig.colorbar(cax)
ticks = np.arange(0,len(df_train.keys()),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(df_train.keys(),rotation='vertical')
ax.set_yticklabels(df_train.keys())

print(Y_train.max())
plt.show()
