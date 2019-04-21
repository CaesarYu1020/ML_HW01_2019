from keras import models
from keras import layers
from keras.datasets import boston_housing
import numpy as np

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean # 减去均值
std = train_data.std(axis=0) # 特征标准差
train_data /= std
test_data -= mean #测试集处理：使用训练集的均值和标准差；不用重新计算
test_data /= std
print(train_data)
def build_model():
    model = models.Sequential()

    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

k = 10
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples] # 划分出验证集部分
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)* num_val_samples:] ],axis=0) # 将训练集拼接到一起
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)* num_val_samples:] ],axis=0)

    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=200,verbose=2)#模型训练silent模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0) # 验证集上评估
    all_scores.append(val_mae)
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets) # score 2.5532484335057877
print(test_mae_score)
print(test_mse_score)
print(all_scores)