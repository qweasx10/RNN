import pandas as pd
import numpy as np
import tensorflow as tf
from google.colab import files
uploaded = files.upload()
import matplotlib.pyplot as plt  
f = pd.read_csv('bus (1).csv')
f.info()
X = f[['lat','lon']]
from sklearn.model_selection import train_test_split
train, test = train_test_split(X, shuffle=False, test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.fit_transform(test)
train_sc_df = pd.DataFrame(train_sc, columns=['lat','lon'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['lat','lon'], index=test.index)
train_sc_df.columns = train.columns 
test_sc_df.columns = test.columns 
column_list = list(train_sc_df)


for s in reversed(range(1, 51)):
    tmp_train = train_sc_df[column_list].shift(s) 
    tmp_test = test_sc_df[column_list].shift(s) 
    tmp_train.columns = "shift_" + tmp_train.columns + "_" + str(-(s-51)) 
    tmp_test.columns = "shift_" + tmp_test.columns + "_" + str(-(s-51)) 
    train_sc_df[tmp_train.columns] = train_sc_df[column_list].shift(s) 
    test_sc_df[tmp_test.columns] = test_sc_df[column_list].shift(s)  
    X_train = train_sc_df.dropna().drop(['lat','lon'], axis=1)
y_train = train_sc_df.dropna()[['lat','lon']]
X_test = test_sc_df.dropna().drop(['lat','lon'], axis=1)
y_test = test_sc_df.dropna()[['lat','lon']]
X_train = X_train.values
X_test= X_test.values
y_train = y_train.values
y_test = y_test.values
print(X_test.shape)
print(y_train.shape)
X_train_t = X_train.reshape(X_train.shape[0], 50, 2)
X_test_t = X_test.reshape(X_test.shape[0], 50, 2)


print("최종 DATA")
print(X_train_t.shape)
print(X_test_t.shape)
print(y_train.shape)
print(y_test.shape)

from keras.layers import LSTM 
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation, Dropout
from keras import optimizers



model = Sequential()
model.add(LSTM(50, batch_input_shape=(80,50,2), stateful=True, return_sequences=True))
model.add(LSTM(50, batch_input_shape=(80,50,2), stateful=True, return_sequences=True))
model.add(LSTM(50, batch_input_shape=(80,50,2), stateful=True))
model.add(Dense(50,activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(2))
    
    
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(X_train_t, y_train, epochs = 1, batch_size = 80)
print('## training loss and acc ##')
print(hist.history['loss'])
print(hist.history['accuracy'])

scores = model.evaluate(X_test_t, y_test)
model.reset_states()
print('## evaluation loss and metrics ##')
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

xhat = X_test_t[0:1]
yhat = model.predict(xhat)
print('## yhat ##')
print(yhat)
    
    
