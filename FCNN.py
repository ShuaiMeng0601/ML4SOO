## FCNN framework
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from .func import L2_loss
from tensorflow.keras.models import save_model

train_data = np.load('train_data.npz')
acc_train = train_data['acc_train']
aabw_train = train_data['aabw_train']
aaiw_train = train_data['aaiw_train']
ssh_train = train_data['ssh_train']
obp_train = train_data['obp_train']

val_data = np.load('val_data.npz')
acc_val = val_data['acc_val']
aabw_val = val_data['aabw_val']
aaiw_val = val_data['aaiw_val']
ssh_val = val_data['ssh_val']
obp_val = val_data['obp_val']

test_data = np.load('test_data.npz')
acc_test = test_data['acc_test']
aabw_test = test_data['aabw_test']
aaiw_test = test_data['aaiw_test']
ssh_test = test_data['ssh_test']
obp_test = test_data['obp_test']

train_stacked_ssh_obp = np.zeros((25550, 256, 512, 2), dtype='float32')
train_stacked_ssh_obp[:,:,:,0] = np.squeeze(ssh_train)
train_stacked_ssh_obp[:,:,:,1] = np.squeeze(obp_train)

val_stacked_ssh_obp = np.zeros((3650, 256, 512, 2), dtype='float32')
val_stacked_ssh_obp[:,:,:,0] = np.squeeze(ssh_val)
val_stacked_ssh_obp[:,:,:,1] = np.squeeze(obp_val)

test_stacked_ssh_obp = np.zeros((3650, 256, 512, 2), dtype='float32')
test_stacked_ssh_obp[:,:,:,0] = np.squeeze(ssh_test)
test_stacked_ssh_obp[:,:,:,1] = np.squeeze(obp_test)
keras.backend.clear_session()

two_inputs = keras.Input(shape=(256,512,2)) # 2 represents the # of channels
x = keras.layers.Flatten()(two_inputs)
x = keras.layers.Dense(1024, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(256, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

x = keras.layers.Dense(64, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

hidden = keras.layers.Dense(32, activation='relu')(x)
x = keras.layers.BatchNormalization()(x)

acc_prediction = keras.layers.Dense(1, activation='linear', name='acc')(hidden)
aabw_prediction = keras.layers.Dense(1,activation='linear',name='aabw')(hidden)
aaiw_prediction = keras.layers.Dense(1,activation='linear',name='aaiw')(hidden)

model = keras.Model(two_inputs, [acc_prediction, aabw_prediction, aaiw_prediction])

#Defining optimizers
SGD = keras.optimizers.SGD(lr=0.001, decay=1e-4, momentum=0.9, nesterov=False)
L2=keras.losses.mean_squared_error

#Compiling the model with chosen loss and optimizer
model.compile(optimizer=SGD, loss=[L2,L2,L2], metrics=[L2_loss])

history_1 = model.fit(train_stacked_ssh_obp, [np.array(acc_train),np.array(aabw_train), np.array(aaiw_train)], epochs=25,
                      batch_size=64,
                      validation_data=(val_stacked_ssh_obp, [np.array(acc_val),np.array(aabw_val),np.array(aaiw_val)]))
[y_pred_acc,y_pred_aabw,y_pred_aaiw] = model.predict(test_stacked_ssh_obp)
mse_test_3 = model.evaluate(test_stacked_ssh_obp, [np.array(acc_test),np.array(aabw_test),np.array(aaiw_test)])

save_model(model, "FCNN_high.h5")