##### Generate training, testing and validation data

## Intro package and library
# Python ≥3.5
import sys
assert sys.version_info >= (3, 5)

# TensorFlow ≥2.0
import tensorflow as tf
assert tf.__version__ >= "2.0"

import numpy as np
import pandas as pd
import os
import random

import scipy.io
import h5py
from .func import normalize

# better image output
import matplotlib as mpl
import matplotlib.pyplot as plt

# ignore the useless warning (SciPy issue #5998)
import warnings

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# make every results identical
np.random.seed(42)

# create an images folder
PROJECT_ROOT_DIR = "."
IMAGE_FILE = "EddyML_images"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, IMAGE_FILE)
os.makedirs(IMAGES_PATH, exist_ok=True)

warnings.filterwarnings(action="ignore", message="^internal gelsd")


## Import data

chunk_name = 'ACC_AABW_ML_doubleMOC_hires_InstTrans'
chunk2 = h5py.File(chunk_name + '_chunk2-003.mat','r')
chunk3 = h5py.File(chunk_name + '_chunk3-007.mat','r')
chunk4 = h5py.File(chunk_name + '_chunk4-008.mat','r')
chunk5 = h5py.File(chunk_name + '_chunk5-002.mat','r')
chunk6 = h5py.File(chunk_name + '_chunk6-010.mat','r')
chunk7 = h5py.File(chunk_name + '_chunk7-006.mat','r')
chunk8 = h5py.File(chunk_name + '_chunk8-005.mat','r')
chunk9 = h5py.File(chunk_name + '_chunk9-009.mat','r')
chunk10 = h5py.File(chunk_name + '_chunk10-001.mat','r')

for i in range(2, 11):
    exec(f'Taabw_{i} = np.array(chunk{i}["Taabw_chunk"][:])')
    exec(f'Taaiw_{i} = np.array(chunk{i}["Taaiw_chunk"][:])')
    exec(f'Tacc_{i} = np.array(chunk{i}["Tacc_chunk"][:])')
    exec(f'ssh_{i} = np.array(chunk{i}["ssh_chunk"][:])')
    exec(f'obp_{i} = np.array(chunk{i}["obp_chunk"][:])')

Taabw = np.vstack((Taabw_2, Taabw_3, Taabw_4, Taabw_5, Taabw_6, Taabw_7, Taabw_8, Taabw_9, Taabw_10))
Taaiw = np.vstack((Taaiw_2, Taaiw_3, Taaiw_4, Taaiw_5, Taaiw_6, Taaiw_7, Taaiw_8, Taaiw_9, Taaiw_10))
Tacc = np.vstack((Tacc_2, Tacc_3, Tacc_4, Tacc_5, Tacc_6, Tacc_7, Tacc_8, Tacc_9, Tacc_10))
ssh = np.concatenate((ssh_2, ssh_3, ssh_4, ssh_5, ssh_6, ssh_7, ssh_8, ssh_9, ssh_10),axis=0)
obp = np.concatenate((obp_2, obp_3, obp_4, obp_5, obp_6, obp_7, obp_8, obp_9, obp_10),axis=0)

ssh_average = (np.mean(ssh,0)).T
obp_average = (np.mean(obp,0)).T
Taabw_average = np.mean(Taabw)
Taaiw_average = np.mean(Taaiw)
Tacc_average = np.mean(Tacc)

SSH = ssh.T
OBP = obp.T

# remove the long-term trend
ssh_trend_remove = np.subtract(SSH, np.expand_dims(ssh_average, -1))
obp_trend_remove = np.subtract(OBP, np.expand_dims(obp_average, -1))
Taabw_trend_remove = np.subtract(Taabw, np.expand_dims(Taabw_average, -1))
Taaiw_trend_remove = np.subtract(Taaiw, np.expand_dims(Taaiw_average, -1))
Tacc_trend_remove = np.subtract(Tacc, np.expand_dims(Tacc_average, -1))


ssh_nor = normalize(ssh_trend_remove.astype(np.float32))
obp_nor = normalize(obp_trend_remove.astype(np.float32))
Taabw_nor = normalize(Taabw_trend_remove.astype(np.float32))
Taaiw_nor = normalize(Taaiw_trend_remove.astype(np.float32))
Tacc_nor = normalize(Tacc_trend_remove.astype(np.float32))

# select one chunk for validation and generate training and testing data
val_index=random.sample(range(0,9), 1)[0]
index_list = list(range(0,9))
index_list.pop(val_index)
test_index=random.sample(index_list, 1)[0]

exclude_idx = np.append(np.arange(val_index*3650,(val_index+1)*3650),
                        np.arange(test_index*3650, (test_index+1)*3650))
ssh_val = (ssh_nor[:,:,val_index*3650:(val_index+1)*3650]).T
ssh_test = (ssh_nor[:,:,test_index*3650:(test_index+1)*3650]).T
ssh_train = (np.delete(ssh_nor, exclude_idx, axis = -1)).T
obp_val = (obp_nor[:,:,val_index*3650:(val_index+1)*3650]).T
obp_test = (obp_nor[:,:,test_index*3650:(test_index+1)*3650]).T
obp_train = (np.delete(obp_nor, exclude_idx, axis = -1)).T

aabw_val = Taabw_nor[val_index*3650:(val_index+1)*3650]
aabw_test = Taabw_nor[test_index*3650:(test_index+1)*3650]
aabw_train = np.delete(Taabw_nor, exclude_idx, axis = 0)
acc_val = Tacc_nor[val_index*3650:(val_index+1)*3650]
acc_test = Tacc_nor[test_index*3650:(test_index+1)*3650]
acc_train = np.delete(Tacc_nor, exclude_idx, axis = 0)
aaiw_val = Taaiw_nor[val_index*3650:(val_index+1)*3650]
aaiw_test = Taaiw_nor[test_index*3650:(test_index+1)*3650]
aaiw_train = np.delete(Taaiw_nor, exclude_idx, axis = 0)

ssh_val = np.expand_dims(ssh_val, -1)
ssh_test = np.expand_dims(ssh_test, -1)
ssh_train = np.expand_dims(ssh_train, -1)
obp_val = np.expand_dims(obp_val, -1)
obp_test = np.expand_dims(obp_test, -1)
obp_train = np.expand_dims(obp_train, -1)

np.savez_compressed('train_data', ssh_train = ssh_train, obp_train = obp_train,
                   acc_train = acc_train, aabw_train = aabw_train, aaiw_train = aaiw_train)

np.savez_compressed('test_data', ssh_test = ssh_test, obp_test = obp_test,
                   acc_test = acc_test, aabw_test = aabw_test, aaiw_test = aaiw_test)

np.savez_compressed('val_data', ssh_val = ssh_val, obp_val = obp_val,
                   acc_val = acc_val, aabw_val = aabw_val, aaiw_val = aaiw_val)
