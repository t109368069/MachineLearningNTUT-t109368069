import random as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# transfer zipcode to one hot encoding
def getData(path):
    df = pd.read_csv(path)
    Y = np.array([])
    zip = pd.get_dummies(df['zipcode'])#轉成one hot encode
    df = df.join(zip)
    #df = df.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day', 'lat', 'long'])
    #這樣才會刪
    df.drop(columns=['id', 'zipcode', 'sale_yr', 'sale_month', 'sale_day', 'lat', 'long'])
    #刪掉不要的
    dataset = np.array(df.values)#剩下要得存成陣列
    if "price" in df.columns:#判斷有沒有價錢(只有test沒有價錢)
        Y = dataset[:, 1]
        dataset = np.delete(dataset, 1, 1)
    dataset = np.delete(dataset, 0, 1)
    return dataset, Y

#用檔名(.csv)呼叫getData，回傳的寫到x y函數裡面(./跟main.py同路徑下的檔案)
# Read training dataset into X and Y
X_train, Y_train = getData('./train-v3.csv')

# Read validation dataset into X and Y
X_valid, Y_valid = getData('./valid-v3.csv')

# Read test dataset into X
X_test, _ = getData('./test-v3.csv')


def normalize(train,valid,test):
	tmp=train
	mean=tmp.mean(axis=0)
	std=tmp.std(axis=0)
	train=(train-mean)/std
	valid=(valid-mean)/std
	test=(test-mean)/std
	return train,valid,test

X_train,X_valid,X_test=normalize(X_train,X_valid,X_test)

from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(40, input_dim=X_train.shape[1]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss='mae')

#驗證(epochs-迭代)
testloss = model.fit(X_train, Y_train, batch_size=30, epochs=150, validation_data=(X_valid, Y_valid))

#畫圖
plt.plot(testloss.history['loss'])
plt.plot(testloss.history['val_loss'])
plt.show()

Y_predict = model.predict(X_test)#把測試資料丟入model把結果存在Y_predict

n = len(Y_predict) + 1
for i in range(1, n):
	b = np.arange(1, n, 1)
	b = np.transpose([b])
	Y = np.column_stack((b, Y_predict))

np.savetxt('./test.csv', Y, delimiter=',', fmt='%i')
