import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
from sklearn import preprocessing

def maxmin(file_dir,input_name, label_name):
    file_name = get_files(file_dir)
    sample_num = len(file_name)

    inputmaxL, inputminL = [], []
    labelmaxL, labelminL = [],[]
    for index in range(sample_num):
        data = np.load(file_name[index])
        data_x = data[input_name]
        data_y = data[label_name]
        data_shape = np.shape(data_x)
        input_max = []
        input_min=[]
        for i in range(data_shape[-1]):
            input_max.append(np.max(abs(data_x[...,i])))
            input_min.append(np.min(abs(data_x[...,i])))
        inputmaxL.append(input_max)
        inputminL.append(input_min)
        label_max = np.max(data_y)
        label_min = np.min(data_y)
        labelmaxL.append(label_max)
        labelminL.append(label_min)
        if (index + 1) % 100 == 0:
            print(index)
    data_inputmax = np.asarray(inputmaxL)
    data_inputmin = np.asarray(inputminL)
    data_labelmax = np.asarray(labelmaxL)
    data_labelmin = np.asarray(labelminL)

    return data_inputmax,data_inputmin,data_labelmax,data_labelmin
def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 文件夹下的所有文件名
    file_name = []
    # 载入数据路径并写入标签值
    # list_dir = os.listdir(file_dir)
    for file in os.listdir(file_dir):
        file_name.append(file_dir+file)
    return file_name
def get_data(file_dir,input_name, label_name, sample_num=None,is_test=False):
    file_name = get_files(file_dir)
    if is_test == False:
        sample_num = len(file_name)
    input_sequence, label_sequence = [], []
    for index in range(sample_num):
        data = np.load(file_name[index])
        # data_x = data[input_name]
        # data_y = data[label_name]
        data_x = data[input_name].astype(np.float32)
        data_y = data[label_name].astype(np.float32)
        input_sequence.append(data_x)
        label_sequence.append(data_y)
        if (index + 1) % 100 == 0:
            print(index)
    data_input = np.asarray(input_sequence)
    data_label = np.asarray(label_sequence)

    return data_input, data_label
def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch
def norm(data_train,data_test,version):
    if version == 1:
        ## 1.自定义max-min归一化
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_shape = np.shape(data_train)
        for i in range(data_shape[-1]):
            train_max = np.max(abs(data_train[:,:,:,i]))
            train_min = np.min(abs(data_train[:,:,:,i]))
            data_train[:, :, :, i] = (data_train[:,:,:,i] -train_min)/ (train_max - train_min)
            data_test[:, :, :, i] = (data_test[:,:,:,i] -train_min)/ (train_max - train_min)
        ## 2.processing MinMax归一化
        # min_max_scaler = preprocessing.MinMaxScaler()
        # data_norm = min_max_scaler.fit_transform(data_pre)
        # data_norm = np.zeros(data_pre.shape, dtype=np.float64)
        ## 3.cv归一化
        # cv.normalize(data_pre, data_norm, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_64F)
    if version == 2:
        ## 1.自定义标准化
        data_train = np.array(data_train)
        data_test = np.array(data_test)
        data_shape = np.shape(data_train)
        for i in range(data_shape[-1]):
            train_mean = np.mean(data_train[:,:,:,i])
            train_std = np.std(data_train[:,:,:,i])
            data_train[:, :, :, i] = (data_train[:,:,:,i] - train_mean) / train_std
            data_test[:, :, :, i] = (data_test[:,:,:,i] - train_mean) / train_std
        ## 2.processing scale标准化
        # data_norm = preprocessing.scale(data_pre)
        ## 3.processing standardscaler标准化
        # data_norm = preprocessing.StandardScaler().fit(data_pre)
    return data_train,data_test
