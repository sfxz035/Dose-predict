import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv
from sklearn import preprocessing


def get_files(file_dir):
    # file_dir: 文件夹路径
    # return: 文件夹下的所有文件名
    file_name = []
    # 载入数据路径并写入标签值
    # list_dir = os.listdir(file_dir)
    for file in os.listdir(file_dir):
        file_name.append(file_dir+file)
    return file_name
def get_data(file_dir,input_name, label_name, is_norm = False):
    file_name = get_files(file_dir)
    sample_num = len(file_name)
    input_sequence, label_sequence = [], []
    for index in range(sample_num):
        data = np.load(file_name[index])
        data_x = data[input_name]
        data_y = data[label_name]
        # if is_norm == True:
        #     for i in range(6):
        #         data_x[:,:,i] = norm(data_x[:,:,i],version=3)
        #     data_y = norm(data_y,version=3)

        # for i in range(data_x.shape[2]):
        #     plt.imshow(data_x[:,:,i],cmap='gray')
        #     plt.show()
        # plt.imshow(data_y, cmap='gray')
        # plt.show()
        input_sequence.append(data_x)
        label_sequence.append(data_y)
        if (index+1)%100 == 0:
            print(index)
    data_input = np.asarray(input_sequence)
    data_label = np.asarray(label_sequence)

    # for i in range(6):
    #     plt.imshow(data_input[1,:,:,i],cmap='gray')
    #     plt.show()
    # plt.imshow(data_label[1,:,:], cmap='gray')
    # plt.show()
    return data_input,data_label
def random_batch(x_data,y_data,batch_size):
    rnd_indices = np.random.randint(0, len(x_data), batch_size)
    x_batch = x_data[rnd_indices][:]
    y_batch = y_data[rnd_indices][:]
    return x_batch, y_batch
def norm(data_train,data_test,version):
    if version == 1:
        ## 1.自定义max-min归一化
        train_max = np.max(abs(data_train))
        train_min = np.min(abs(data_train))

        train_norm = (data_train -train_min)/ (train_max - train_min)
        test_norm = (data_test -train_min)/ (train_max - train_min)
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
        train_mean = np.mean(data_train)
        train_std = np.std(data_train)
        train_norm = (data_train - train_mean) / train_std
        test_norm = (data_test - train_mean) / train_std
        ## 2.processing scale标准化
        # data_norm = preprocessing.scale(data_pre)
        ## 3.processing standardscaler标准化
        # data_norm = preprocessing.StandardScaler().fit(data_pre)
    return train_norm,test_norm
