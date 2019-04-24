import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import Unet
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)
def test():
    savepath = './libSaveNet/save_unet4/conv_unet299999.ckpt-done'
    Test_dir = './data/train/data.0.71.npz'
    data_test = np.load(Test_dir)
    Test_input = data_test['RTSt']
    data_train = np.array(Test_input)
    train_mean = np.mean(data_train)
    train_std = np.std(data_train)
    Test_input = (data_train - train_mean) / train_std
    length = Test_input.shape[1]
    Test_input = np.expand_dims(Test_input, 0)

    Test_label = data_test['RTDose']
    Test_label = np.array(Test_label)
    test_mean = np.mean(Test_label)
    test_std = np.std(Test_label)
    Test_label = (Test_label - test_mean) / test_std

    Test_label = np.expand_dims(Test_label, -1)
    Test_label = np.expand_dims(Test_label, 0)

    x = tf.placeholder(tf.float32,shape = [None,512,512, 6])
    y_ = tf.placeholder(tf.float32,shape = [None,512,512,1])
    dropout_value = tf.placeholder(tf.float32)  # 参与节点的数目百分比
    is_training = tf.placeholder(tf.bool)

    y = Unet.net(x, len=length,is_training=is_training,dropout_value=dropout_value)

    loss = tf.reduce_mean(tf.square(y - y_))

    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    tf.global_variables_initializer().run()
    saver.restore(sess, savepath)
    output = sess.run(y, feed_dict={x: Test_input,is_training:False,dropout_value:0})
    output = np.squeeze(output)
    loss_train = sess.run(loss, feed_dict={x: Test_input, y_: Test_label,is_training:False,dropout_value:0})
    print('loss_train: %g' % (loss_train))
    # for i in range(6):
    #     plt.imshow(Test_input[0,:,:,i])
    #     plt.show()
    ca = Test_label[0,:,:,0]-output
    plt.imshow(Test_label[0,:,:,0])
    plt.show()
    plt.imshow(output)
    plt.show()
    plt.imshow(ca)
    plt.show()
test()