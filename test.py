import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networks.U_net as U_net
import dataset
import cv2 as cv
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)

trainfile_dir = './data/train/'
input_name = 'RTSt'
label_name = 'RTDose'
channel = 6
def test():
    savepath = './libSaveNet/save_unet5/conv_unet79999.ckpt-done'
    ## batch
    # rnd_indices = [20,45,56,26,35,5,78]
    # inputsuqen = []
    # labelsuqen = []
    # for i in range(batch_size):
    #     path = './data/train/data.0.'+str(rnd_indices[i])+'.npz'
    #     data = np.load(path)
    #     data_input = data['RTSt']
    #     data_label = data['RTDose']
    #     inputsuqen.append(data_input)
    #     labelsuqen.append(data_label)
    # Test_input = np.asarray(inputsuqen)


    ## 单个
    Test_dir = './data/train/data.0.103.npz'
    data_test = np.load(Test_dir)
    Test_input = data_test['RTSt']
    data_train = np.array(Test_input)


    Test_label = data_test['RTDose']
    Test_label = np.array(Test_label)



    #归一化
    # input_max,input_min,label_max,label_min = dataset.maxmin(trainfile_dir,input_name,label_name)
    input_max = [255,255,255,255,255,255,4095]
    input_min = [0,0,0,0,0,0,24]
    label_max = 1315878.25
    label_min = 0
    for i in range(channel):
        # train_max = np.max(input_max[...,i])
        # train_min = np.min(input_min[...,i])
        train_max = np.max(input_max[i])
        train_min = np.min(input_min[i])
        data_train[..., i] = (data_train[..., i] - train_min) / (train_max - train_min)
    Test_input= data_train
    labmax = np.max(label_max)
    labmin = np.min(label_min)
    Test_label = (Test_label - labmin) / (labmax - labmin)


    #标准化
    # train_mean = np.mean(data_train)
    # train_std = np.std(data_train)
    # Test_input = (data_train - train_mean) / train_std
    # test_mean = np.mean(Test_label)
    # test_std = np.std(Test_label)
    # Test_label = (Test_label - test_mean) / test_std


    Test_input = np.expand_dims(Test_input, 0)
    Test_label = np.expand_dims(Test_label, -1)
    Test_label = np.expand_dims(Test_label, 0)

    x = tf.placeholder(tf.float32,shape = [1,512,512, channel])
    y_ = tf.placeholder(tf.float32,shape = [1,512,512,1])

    y = U_net.inference(x, dropout_value=0)

    loss = tf.reduce_mean(tf.square(y - y_))

    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    output = sess.run(y, feed_dict={x: Test_input})
    loss_train = sess.run(loss, feed_dict={x: Test_input, y_: Test_label})
    print('loss_train: %g' % (loss_train))

    # out = (output[0,:,:,0]*255).astype(np.uint8)
    label = (Test_label[0,:,:,0])
    ca = abs(Test_label[0,:,:,0]-output[0,:,:,0])
    # cv.namedWindow('label',0)
    # cv.resizeWindow('label',500,500)
    # cv.imshow('label',label)
    # cv.namedWindow('output',0)
    # cv.resizeWindow('output',500,500)
    # cv.imshow('output',out)
    # cv.namedWindow('ca',0)
    # cv.resizeWindow('ca',500,500)
    # cv.imshow('ca',ca)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    plt.imshow(label)
    plt.show()
    plt.imshow(output[0,:,:,0])
    plt.show()
    plt.imshow(ca)
    plt.show()

if __name__ == '__main__':
    test()