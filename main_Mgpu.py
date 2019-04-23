import tensorflow as tf
import os
import numpy as np
import dataset
import Unet
import networks.U_net as U_net
import matplotlib.pyplot as plt
import scipy.io as sio
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '1'   #指定第一块GPU可用
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存
sess = tf.InteractiveSession(config = config)

epoch = 2000000
batch_size = 7
learning_rate = 0.0001
savenet_path = './libSaveNet/save_unet/'
trainfile_dir = './data/train/'
testfile_dir = './data/test/'
input_name = 'RTSt'
label_name = 'RTDose'
num_gpus = 2
# #一次读取
# x_data,y_data = dataset.get_data(trainfile_dir, input_name, label_name,is_norm=True)
# x_train = x_data[:-231,:,:,:]
# y_train = y_data[:-231,:,:,:]
# x_test = x_data[-231:,:,:,:]
# y_test = y_data[-231:,:,:,:]
## 分别读取
x_train,y_train = dataset.get_data(trainfile_dir, input_name, label_name)

x_test,y_test = dataset.get_data(testfile_dir, input_name, label_name)

x_train,x_test = dataset.norm(x_train,x_test,version=2)
y_train,y_test = dataset.norm(y_train,y_test,version=2)
# for i in range(6):
#     plt.imshow(x_train[4,:,:,i],cmap='gray')
#     plt.show()
# plt.imshow(y_train[4,:,:], cmap='gray')
# plt.show()
length = x_train.shape[1]
y_train = np.expand_dims(y_train,-1)
y_test = np.expand_dims(y_test,-1)


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expend_g = tf.expand_dims(g, 0)
            grads.append(expend_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
def train():
    with tf.device("/cpu:0"):
        global_step = tf.get_variable('global_step', [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)
        # num_batches_per_epoch
        tower_grads = []
        x = tf.placeholder(tf.float32, shape=[None, 512, 512, 6])
        y_ = tf.placeholder(tf.float32, shape=[None, 512, 512, 1])
        opt = tf.train.AdamOptimizer(learning_rate)
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(num_gpus):
                with tf.device("/gpu:%d" % i):
                    with tf.name_scope("tower_%d" % i):
                        _x = x[i * batch_size:(i + 1) * batch_size]
                        _y = y_[i * batch_size:(i + 1) * batch_size]
                        y = U_net.inference(_x,is_training=True)
                        tf.get_variable_scope().reuse_variables()
                        loss = tf.reduce_mean(tf.square(_y - y))
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
                        # if i == 0:
                        #     logits_test = U_net.inference(_x,is_training=False)
                            # correct_prediction = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        grads = average_gradients(tower_grads)
        train_op = opt.apply_gradients(grads)
        saver = tf.train.Saver(tf.all_variables, write_version=tf.train.SaverDef.V2, max_to_keep=8)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))     ##自动选择一个存在并且支持的设备运行
        sess.run(init)
        count, m = 0, 0
        for ep in range(epoch):
            batch_idxs = len(x_train) // batch_size
            for step in range(batch_idxs):
                batch_input, batch_labels = dataset.random_batch(x_train,y_train,batch_size)
                start_time = time.time()
                sess.run(train_op, feed_dict={x: batch_input, y_: batch_labels})
                duration = time.time()-start_time
                if count % 50 == 0:
                    num_examples_per_step = batch_size * num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / num_gpus
                    batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, batch_size)
                    loss_value = sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test})
                    print('step %d, loss=%.2f(%.1f examples/sec;%.3f sec/batch)'
                          % (step, loss_value, examples_per_sec, sec_per_batch))
                if (count + 1) % 20000 == 0:
                    saver.save(sess, os.path.join(savenet_path, 'conv_unet%d.ckpt-done' % (count)))

if __name__ == '__main__':
    train()