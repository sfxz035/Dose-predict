import tensorflow as tf
import os
import numpy as np
import dataset
import Unet
import scipy.io as sio

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
x_train,y_train = dataset.get_data(trainfile_dir, input_name, label_name,is_norm=True)
length = x_train.shape[1]
y_train = np.expand_dims(y_train,-1)
x_test,y_test = dataset.get_data(testfile_dir, input_name, label_name,is_norm=True)
y_test = np.expand_dims(y_test,-1)

def train():
    x = tf.placeholder(tf.float32,shape = [None,512,512, 6])
    y_ = tf.placeholder(tf.float32,shape = [None,512,512,1])
    is_training = tf.placeholder(tf.bool)
    dropout_value = tf.placeholder(tf.float32)  # 参与节点的数目百分比


    y = Unet.net(x,len=length,is_training=is_training,dropout_value=dropout_value)

    loss = tf.reduce_mean(tf.square(y - y_))

    summary_op = tf.summary.scalar('trainloss', loss)
    summary_op2 = tf.summary.scalar('testloss', loss)
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    variables_to_restore = []
    for v in tf.global_variables():
        variables_to_restore.append(v)
    saver = tf.train.Saver(variables_to_restore, write_version=tf.train.SaverDef.V2, max_to_keep=None)
    writer = tf.summary.FileWriter('./my_graph/train', sess.graph)
    writer2 = tf.summary.FileWriter('./my_graph/test')
    tf.global_variables_initializer().run()
    # last_file = tf.train.latest_checkpoint(savenet_path)
    # if last_file:
    #     tf.logging.info('Restoring model from {}'.format(last_file))
        # saver.restore(sess, last_file)

    count, m = 0, 0
    for ep in range(epoch):
        batch_idxs = len(x_train) // batch_size
        for idx in range(batch_idxs):
            # batch_input = x_train[idx * batch_size: (idx + 1) * batch_size]
            # batch_labels = y_train[idx * batch_size: (idx + 1) * batch_size]
            batch_input, batch_labels = dataset.random_batch(x_train,y_train,batch_size)
            sess.run(train_step, feed_dict={x: batch_input, y_: batch_labels,is_training:True,dropout_value:0.85})
            count += 1
            # print(count)
            if count % 50 == 0:
                m += 1
                batch_input_test, batch_labels_test = dataset.random_batch(x_test, y_test, batch_size)
                # batch_input_test = x_test[0 : batch_size]
                # batch_labels_test = y_test[0 : batch_size]
                loss1 = sess.run(loss, feed_dict={x: batch_input,y_: batch_labels,is_training:False,dropout_value:1})
                loss2 = sess.run(loss, feed_dict={x: batch_input_test, y_: batch_labels_test,is_training:False,dropout_value:1})
                print("Epoch: [%2d], step: [%2d], train_loss: [%.8f]" \
                      % ((ep + 1), count, loss1), "\t", 'test_loss:[%.8f]' % (loss2))
                writer.add_summary(sess.run(summary_op, feed_dict={x: batch_input, y_: batch_labels,is_training:False,dropout_value:1}), m)
                writer2.add_summary(sess.run(summary_op2, feed_dict={x: batch_input_test,
                                                                     y_: batch_labels_test,is_training:False,dropout_value:1}), m)
            if (count + 1) % 20000 == 0:
                saver.save(sess, os.path.join(savenet_path, 'conv_unet%d.ckpt-done' % (count)))


if __name__ == '__main__':
    train()