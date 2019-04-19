import tensorflow as tf
import math
from tensorflow import keras
EPS = 10e-5
# batch_size = 10
input_us = 512
FullConect1 = 512
FullConect2 = 512
FullConect3 = 512

### level-1
conv1_size = 3
inp1_channel = 6
out1_channel = 24

conv2_size = 3
inp2_channel = 24
out2_channel = 24
### level-2
conv3_size = 3
inp3_channel = 24
out3_channel = 48

conv4_size = 3
inp4_channel = 48
out4_channel = 48

### level-3
conv5_size = 3
inp5_channel = 48
out5_channel = 96

conv6_size = 3
inp6_channel = 96
out6_channel = 96

### level-4
conv7_size = 3
inp7_channel = 96
out7_channel = 192

conv8_size = 3
inp8_channel = 192
out8_channel = 192

### level-5
conv9_size = 3
inp9_channel = 192
out9_channel = 384

conv10_size = 3
inp10_channel = 384
out10_channel = 384

### level-6
conv11_size = 3
inp11_channel = 384
out11_channel = 768

conv12_size = 3
inp12_channel = 768
out12_channel = 768

### level-7(bottom)
conv13_size = 3
inp13_channel = 768
out13_channel = 1536

conv14_size = 3
inp14_channel = 1536
out14_channel = 1536

transconv1 = 3
up_out1_channel = 768
up_inp1_channel = 1536

### level-8
conv15_size = 3
inp15_channel = 1536
out15_channel = 768

conv16_size = 3
inp16_channel = 768
out16_channel = 768

transconv2 = 3
up_out2_channel = 384
up_inp2_channel = 768

### level-9
conv17_size = 3
inp17_channel = 768
out17_channel = 384

conv18_size = 3
inp18_channel = 384
out18_channel = 384

transconv3 = 3
up_out3_channel = 192
up_inp3_channel = 384

### level-10
conv19_size = 3
inp19_channel = 384
out19_channel = 192

conv20_size = 3
inp20_channel = 192
out20_channel = 192

transconv4 = 3
up_out4_channel = 96
up_inp4_channel = 192


### level-11
conv21_size = 3
inp21_channel = 192
out21_channel = 96

conv22_size = 3
inp22_channel = 96
out22_channel = 96

transconv5 = 3
up_out5_channel = 48
up_inp5_channel = 96


### level-12
conv23_size = 3
inp23_channel = 96
out23_channel = 48

conv24_size = 3
inp24_channel = 48
out24_channel = 48

transconv6 = 3
up_out6_channel = 24
up_inp6_channel = 48


### addtional conv
conv25_size = 3
inp25_channel = 48
out25_channel = 24

conv26_size = 3
inp26_channel = 24
out26_channel = 24

conv27_size = 3
inp27_channel = 24
out27_channel = 6  #24

conv28_size = 3
inp28_channel = 6   #24
out28_channel = 2   #24

conv29_size = 3
inp29_channel = 2    #24
out29_channel = 1    #12

# conv30_size = 3
# inp30_channel = 12
# out30_channel = 6
#
# conv31_size = 3
# inp31_channel = 6
# out31_channel = 3
#
# conv32_size = 3
# inp32_channel = 3
# out32_channel = 2
#
# conv33_size = 3
# inp33_channel = 2
# out33_channel = 1

def batch_norm(x, is_training, eps=EPS, decay=0.9, affine=True, name='BatchNorm2d'):
    from tensorflow.python.training.moving_averages import assign_moving_average

    with tf.variable_scope(name):
        params_shape = x.shape[-1:]
        moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                     trainable=False)

        def mean_var_with_update():
            mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
            with tf.control_dependencies([
                assign_moving_average(moving_mean, mean_this_batch, decay),
                assign_moving_average(moving_var, variance_this_batch, decay)
            ]):
                return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

        mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
        if affine:  # 如果要用beta和gamma进行放缩
            beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                               variance_epsilon=eps)
        else:
            normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                               variance_epsilon=eps)
        return normed


def int_w(shape,name):
    weights = tf.get_variable(
        name=name, shape=shape,dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1)
        # initializer=tf.contrib.layers.xavier_initializer()
    )
    return weights

def int_b(shape,name):
    bais = tf.get_variable(
        name=name, shape=shape, initializer=tf.constant_initializer(0.0)
    )
    return bais

def net(input,len,dropout_value):
    batch_size = tf.shape(input)[0]
    len_layer1 = len
    len_layer2 = math.ceil(len_layer1/2)
    len_layer3 = math.ceil(len_layer2/2)
    len_layer4 = math.ceil(len_layer3/2)
    len_layer5 = math.ceil(len_layer4/2)
    len_layer6 = math.ceil(len_layer5/2)
    ### layer1
       #conv1
    w1 = int_w(shape=[conv1_size,conv1_size,inp1_channel,out1_channel],name='W1')
    conv1 = tf.nn.conv2d(
        input, w1, strides=[1, 1, 1, 1], padding='SAME',name='conv1'
    )
    relu1 = tf.nn.relu(tf.layers.batch_normalization(conv1,training=True))
    # b1 = int_b(shape=[out1_channel],name='b1')
    # relu1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
    relu1 = tf.nn.dropout(relu1, keep_prob=dropout_value)  # dropout

       #conv2
    w2 = int_w(shape=[conv2_size,conv2_size,inp2_channel,out2_channel],name='W2')
    conv2 = tf.nn.conv2d(
        relu1,w2,strides=[1, 1, 1, 1],padding='SAME',name='conv2'
    )
    relu2 = tf.nn.relu(tf.layers.batch_normalization(conv2,training=True))
    # b2 = int_b(shape=[out2_channel],name='b2')
    # relu2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
    pool1 = tf.nn.max_pool(
        relu2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME'
    )

    ### layer2
    # conv1
    w3 = int_w(shape=[conv3_size,conv3_size,inp3_channel,out3_channel],name='W3')
    conv3 = tf.nn.conv2d(
        pool1, w3, strides=[1, 1, 1, 1], padding='SAME',name='conv3'
    )
    relu3 = tf.nn.relu(tf.layers.batch_normalization(conv3,training=True))
    # b3 = int_b(shape=[out3_channel],name='b3')
    # relu3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))
    relu3 = tf.nn.dropout(relu3, keep_prob=dropout_value)  # dropout

       #conv2
    w4 = int_w(shape=[conv4_size,conv4_size,inp4_channel,out4_channel],name='W4')
    conv4 = tf.nn.conv2d(
        relu3,w4,strides=[1, 1, 1, 1],padding='SAME',name='conv4'
    )
    relu4 = tf.nn.relu(tf.layers.batch_normalization(conv4,training=True))
    # b4 = int_b(shape=[out4_channel],name='b4')
    # relu4 = tf.nn.relu(tf.nn.bias_add(conv4, b4))
    pool2 = tf.nn.max_pool(
        relu4,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME'
    )

    ### layer3
    # conv1
    w5 = int_w(shape=[conv5_size, conv5_size, inp5_channel, out5_channel], name='W5')
    conv5 = tf.nn.conv2d(
        pool2, w5, strides=[1, 1, 1, 1], padding='SAME', name='conv5'
    )
    relu5 = tf.nn.relu(tf.layers.batch_normalization(conv5,training=True))
    # b5 = int_b(shape=[out5_channel],name='b5')
    # relu5 = tf.nn.relu(tf.nn.bias_add(conv5, b5))
    relu5 = tf.nn.dropout(relu5, keep_prob=dropout_value)  # dropout

    # conv2
    w6 = int_w(shape=[conv6_size, conv6_size, inp6_channel, out6_channel], name='W6')
    conv6 = tf.nn.conv2d(
        relu5, w6, strides=[1, 1, 1, 1], padding='SAME', name='conv6'
    )
    relu6 = tf.nn.relu(tf.layers.batch_normalization(conv6,training=True))
    # b6 = int_b(shape=[out6_channel],name='b6')
    # relu6 = tf.nn.relu(tf.nn.bias_add(conv6, b6))

    pool3 = tf.nn.max_pool(
        relu6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )

    ### layer4
    # conv1
    w7 = int_w(shape=[conv7_size, conv7_size, inp7_channel, out7_channel], name='W7')
    conv7 = tf.nn.conv2d(
        pool3, w7, strides=[1, 1, 1, 1], padding='SAME', name='conv7'
    )
    relu7 = tf.nn.relu(tf.layers.batch_normalization(conv7,training=True))
    # b7 = int_b(shape=[out7_channel],name='b7')
    # relu7 = tf.nn.relu(tf.nn.bias_add(conv7, b7))
    relu7 = tf.nn.dropout(relu7, keep_prob=dropout_value)  # dropout

    # conv2
    w8 = int_w(shape=[conv8_size, conv8_size, inp8_channel, out8_channel], name='W8')
    conv8 = tf.nn.conv2d(
        relu7, w8, strides=[1, 1, 1, 1], padding='SAME', name='conv8'
    )
    relu8 = tf.nn.relu(tf.layers.batch_normalization(conv8,training=True))
    # b8 = int_b(shape=[out8_channel],name='b8')
    # relu8 = tf.nn.relu(tf.nn.bias_add(conv8, b8))


    pool4 = tf.nn.max_pool(
        relu8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )
    ### layer5
    # conv1
    w9 = int_w(shape=[conv9_size, conv9_size, inp9_channel, out9_channel], name='W9')
    conv9 = tf.nn.conv2d(
        pool4, w9, strides=[1, 1, 1, 1], padding='SAME', name='conv9'
    )
    relu9 = tf.nn.relu(tf.layers.batch_normalization(conv9,training=True))
    # b9 = int_b(shape=[out9_channel],name='b9')
    # relu9 = tf.nn.relu(tf.nn.bias_add(conv9, b9))
    relu9 = tf.nn.dropout(relu9, keep_prob=dropout_value)  # dropout

    # conv2
    w10 = int_w(shape=[conv10_size, conv10_size, inp10_channel, out10_channel], name='W10')
    conv10 = tf.nn.conv2d(
        relu9, w10, strides=[1, 1, 1, 1], padding='SAME', name='conv10'
    )
    relu10 = tf.nn.relu(tf.layers.batch_normalization(conv10,training=True))
    # b10 = int_b(shape=[out10_channel],name='b10')
    # relu10 = tf.nn.relu(tf.nn.bias_add(conv10, b10))


    pool5 = tf.nn.max_pool(
        relu10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )

    ### layer6
    # conv1
    w11 = int_w(shape=[conv11_size, conv11_size, inp11_channel, out11_channel], name='W11')
    conv11 = tf.nn.conv2d(
        pool5, w11, strides=[1, 1, 1, 1], padding='SAME', name='conv11'
    )
    relu11 = tf.nn.relu(tf.layers.batch_normalization(conv11,training=True))
    # b11 = int_b(shape=[out11_channel],name='b11')
    # relu11 = tf.nn.relu(tf.nn.bias_add(conv11, b11))
    relu11 = tf.nn.dropout(relu11, keep_prob=dropout_value)  # dropout

    # conv2
    w12 = int_w(shape=[conv12_size, conv12_size, inp12_channel, out12_channel], name='W12')
    conv12 = tf.nn.conv2d(
        relu11, w12, strides=[1, 1, 1, 1], padding='SAME', name='conv12'
    )
    relu12 = tf.nn.relu(tf.layers.batch_normalization(conv12,training=True))
    # b12 = int_b(shape=[out12_channel],name='b12')
    # relu12 = tf.nn.relu(tf.nn.bias_add(conv12, b12))


    pool6 = tf.nn.max_pool(
        relu12, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
    )
    ### layer7
    # conv1
    w13 = int_w(shape=[conv13_size, conv13_size, inp13_channel, out13_channel], name='W13')
    conv13 = tf.nn.conv2d(
        pool6, w13, strides=[1, 1, 1, 1], padding='SAME', name='conv13'
    )
    relu13 = tf.nn.relu(tf.layers.batch_normalization(conv13,training=True))
    # b13 = int_b(shape=[out13_channel],name='b13')
    # relu13 = tf.nn.relu(tf.nn.bias_add(conv13, b13))
    relu13 = tf.nn.dropout(relu13, keep_prob=dropout_value)  # dropout

    # conv2
    w14 = int_w(shape=[conv14_size, conv14_size, inp14_channel, out14_channel], name='W14')
    conv14 = tf.nn.conv2d(
        relu13, w14, strides=[1, 1, 1, 1], padding='SAME', name='conv14'
    )
    relu14 = tf.nn.relu(tf.layers.batch_normalization(conv14,training=True))
    # b14 = int_b(shape=[out14_channel],name='b14')
    # relu14 = tf.nn.relu(tf.nn.bias_add(conv14, b14))


    # up sample
    w1_tran = int_w(shape=[transconv1, transconv1, up_out1_channel, up_inp1_channel], name='w1_tran')
    conv1_tran = tf.nn.conv2d_transpose(
        relu14, w1_tran,output_shape=[batch_size, len_layer6, len_layer6, up_out1_channel],strides=[1, 2, 2, 1], padding='SAME'
    )
    relu1_tran = tf.nn.relu(tf.layers.batch_normalization(conv1_tran,training=True))
    # b1_tran = int_b(shape=[up_out1_channel],name='b1_tran')
    # relu1_tran = tf.nn.relu(tf.nn.bias_add(conv1_tran, b1_tran))


    ### layer8
    # conv1
    concat1 = tf.concat([relu12,relu1_tran],-1)

    # conv1
    w15 = int_w(shape=[conv15_size, conv15_size, inp15_channel, out15_channel], name='W15')
    conv15 = tf.nn.conv2d(
        concat1, w15, strides=[1, 1, 1, 1], padding='SAME', name='conv15'
    )
    relu15 = tf.nn.relu(tf.layers.batch_normalization(conv15,training=True))
    # b15 = int_b(shape=[out15_channel],name='b15')
    # relu15 = tf.nn.relu(tf.nn.bias_add(conv15, b15))
    relu15 = tf.nn.dropout(relu15, keep_prob=dropout_value)  # dropout

    # conv2
    w16 = int_w(shape=[conv16_size, conv16_size, inp16_channel, out16_channel], name='W16')
    conv16 = tf.nn.conv2d(
        relu15, w16, strides=[1, 1, 1, 1], padding='SAME', name='conv16'
    )
    relu16 = tf.nn.relu(tf.layers.batch_normalization(conv16,training=True))
    # b16 = int_b(shape=[out16_channel],name='b16')
    # relu16 = tf.nn.relu(tf.nn.bias_add(conv16, b16))

    # up sample
    w2_tran = int_w(shape=[transconv2, transconv2, up_out2_channel, up_inp2_channel], name='w2_tran')
    conv2_tran = tf.nn.conv2d_transpose(
        relu16, w2_tran, output_shape=[batch_size, len_layer5, len_layer5, up_out2_channel], strides=[1, 2, 2, 1], padding='SAME'
    )
    relu2_tran = tf.nn.relu(tf.layers.batch_normalization(conv2_tran,training=True))
    # b2_tran = int_b(shape=[up_out2_channel],name='b2_tran')
    # relu2_tran = tf.nn.relu(tf.nn.bias_add(conv2_tran, b2_tran))

    ### layer9
    # conv1
    concat2 = tf.concat([relu10, relu2_tran], -1)

    # conv1
    w17 = int_w(shape=[conv17_size, conv17_size, inp17_channel, out17_channel], name='W17')
    conv17 = tf.nn.conv2d(
        concat2, w17, strides=[1, 1, 1, 1], padding='SAME', name='conv17'
    )
    relu17 = tf.nn.relu(tf.layers.batch_normalization(conv17,training=True))
    # b17 = int_b(shape=[out17_channel],name='b17')
    # relu17 = tf.nn.relu(tf.nn.bias_add(conv17, b17))
    relu17 = tf.nn.dropout(relu17, keep_prob=dropout_value)  # dropout

    # conv2
    w18 = int_w(shape=[conv18_size, conv18_size, inp18_channel, out18_channel], name='W18')
    conv18 = tf.nn.conv2d(
        relu17, w18, strides=[1, 1, 1, 1], padding='SAME', name='conv18'
    )
    relu18 = tf.nn.relu(tf.layers.batch_normalization(conv18,training=True))
    # b18 = int_b(shape=[out18_channel],name='b18')
    # relu18 = tf.nn.relu(tf.nn.bias_add(conv18, b18))


    # up sample
    w3_tran = int_w(shape=[transconv3, transconv3, up_out3_channel, up_inp3_channel], name='w3_tran')
    conv3_tran = tf.nn.conv2d_transpose(
        relu18, w3_tran, output_shape=[batch_size, len_layer4, len_layer4, up_out3_channel], strides=[1, 2, 2, 1], padding='SAME'
    )
    relu3_tran = tf.nn.relu(tf.layers.batch_normalization(conv3_tran,training=True))

    # b3_tran = int_b(shape=[up_out3_channel],name='b3_tran')
    # relu3_tran = tf.nn.relu(tf.nn.bias_add(conv3_tran, b3_tran))


    ### layer10
    # conv1
    concat3 = tf.concat([relu8, relu3_tran], -1)

    # conv1
    w19 = int_w(shape=[conv19_size, conv19_size, inp19_channel, out19_channel], name='W19')
    conv19 = tf.nn.conv2d(
        concat3, w19, strides=[1, 1, 1, 1], padding='SAME', name='conv19'
    )
    relu19 = tf.nn.relu(tf.layers.batch_normalization(conv19,training=True))
    # b19 = int_b(shape=[out19_channel],name='b19')
    # relu19 = tf.nn.relu(tf.nn.bias_add(conv19, b19))
    relu19 = tf.nn.dropout(relu19, keep_prob=dropout_value)  # dropout

    # conv2
    w20 = int_w(shape=[conv20_size, conv20_size, inp20_channel, out20_channel], name='W20')
    conv20 = tf.nn.conv2d(
        relu19, w20, strides=[1, 1, 1, 1], padding='SAME', name='conv20'
    )
    relu20 = tf.nn.relu(tf.layers.batch_normalization(conv20,training=True))
    # b20 = int_b(shape=[out20_channel],name='b20')
    # relu20 = tf.nn.relu(tf.nn.bias_add(conv20, b20))


    # up sample
    w4_tran = int_w(shape=[transconv4, transconv4, up_out4_channel, up_inp4_channel], name='w4_tran')
    conv4_tran = tf.nn.conv2d_transpose(
        relu20, w4_tran, output_shape=[batch_size, len_layer3, len_layer3, up_out4_channel], strides=[1, 2, 2, 1], padding='SAME'
    )
    relu4_tran = tf.nn.relu(tf.layers.batch_normalization(conv4_tran,training=True))
    # b4_tran = int_b(shape=[up_out4_channel],name='b4_tran')
    # relu4_tran = tf.nn.relu(tf.nn.bias_add(conv4_tran, b4_tran))

    ### layer11
    # conv1
    concat4 = tf.concat([relu6, relu4_tran], -1)

    # conv1
    w21 = int_w(shape=[conv21_size, conv21_size, inp21_channel, out21_channel], name='W21')
    conv21 = tf.nn.conv2d(
        concat4, w21, strides=[1, 1, 1, 1], padding='SAME', name='conv21'
    )
    relu21 = tf.nn.relu(tf.layers.batch_normalization(conv21,training=True))
    # b21 = int_b(shape=[out21_channel],name='b21')
    # relu21 = tf.nn.relu(tf.nn.bias_add(conv21, b21))
    relu21 = tf.nn.dropout(relu21, keep_prob=dropout_value)  # dropout

    # conv2
    w22 = int_w(shape=[conv22_size, conv22_size, inp22_channel, out22_channel], name='W22')
    conv22 = tf.nn.conv2d(
        relu21, w22, strides=[1, 1, 1, 1], padding='SAME', name='conv22'
    )
    relu22 = tf.nn.relu(tf.layers.batch_normalization(conv22,training=True))
    # b22 = int_b(shape=[out22_channel],name='b22')
    # relu22 = tf.nn.relu(tf.nn.bias_add(conv22, b22))

    # up sample
    w5_tran = int_w(shape=[transconv5, transconv5, up_out5_channel, up_inp5_channel], name='w5_tran')
    conv5_tran = tf.nn.conv2d_transpose(
        relu22, w5_tran, output_shape=[batch_size, len_layer2, len_layer2, up_out5_channel], strides=[1, 2, 2, 1], padding='SAME'
    )
    relu5_tran = tf.nn.relu(tf.layers.batch_normalization(conv5_tran,training=True))
    # b5_tran = int_b(shape=[up_out5_channel],name='b5_tran')
    # relu5_tran = tf.nn.relu(tf.nn.bias_add(conv5_tran, b5_tran))

    ### layer12
    # conv1
    concat5 = tf.concat([relu4, relu5_tran], -1)

    # conv1
    w23 = int_w(shape=[conv23_size, conv23_size, inp23_channel, out23_channel], name='W23')
    conv23 = tf.nn.conv2d(
        concat5, w23, strides=[1, 1, 1, 1], padding='SAME', name='conv23'
    )
    relu23 = tf.nn.relu(tf.layers.batch_normalization(conv23,training=True))
    # b23 = int_b(shape=[out23_channel],name='b23')
    # relu23 = tf.nn.relu(tf.nn.bias_add(conv23, b23))
    relu23 = tf.nn.dropout(relu23, keep_prob=dropout_value)  # dropout

    # conv2
    w24 = int_w(shape=[conv24_size, conv24_size, inp24_channel, out24_channel], name='W24')
    conv24 = tf.nn.conv2d(
        relu23, w24, strides=[1, 1, 1, 1], padding='SAME', name='conv22'
    )
    relu24 = tf.nn.relu(tf.layers.batch_normalization(conv24,training=True))
    # b24 = int_b(shape=[out24_channel],name='b24')
    # relu24 = tf.nn.relu(tf.nn.bias_add(conv24, b24))

    # up sample
    w6_tran = int_w(shape=[transconv6, transconv6, up_out6_channel, up_inp6_channel], name='w6_tran')
    conv6_tran = tf.nn.conv2d_transpose(
        relu24, w6_tran, output_shape=[batch_size, len_layer1, len_layer1, up_out6_channel], strides=[1, 2, 2, 1], padding='SAME'
    )
    relu6_tran = tf.nn.relu(tf.layers.batch_normalization(conv6_tran,training=True))
    # b6_tran = int_b(shape=[up_out6_channel],name='b6_tran')
    # relu6_tran = tf.nn.relu(tf.nn.bias_add(conv6_tran, b6_tran))

    #### addtiona conv
    concat6 = tf.concat([relu2, relu6_tran], -1)
    # conv1
    w25 = int_w(shape=[conv25_size, conv25_size, inp25_channel, out25_channel], name='W25')
    conv25 = tf.nn.conv2d(
        concat6, w25, strides=[1, 1, 1, 1], padding='SAME', name='conv26'
    )
    relu25 = tf.nn.relu(tf.layers.batch_normalization(conv25,training=True))
    # b25 = int_b(shape=[out25_channel],name='b25')
    # relu25 = tf.nn.relu(tf.nn.bias_add(conv25, b25))
    relu25 = tf.nn.dropout(relu25, keep_prob=dropout_value)  # dropout

    # conv2
    w26 = int_w(shape=[conv26_size, conv26_size, inp26_channel, out26_channel], name='W26')
    conv26 = tf.nn.conv2d(
        relu25, w26, strides=[1, 1, 1, 1], padding='SAME', name='conv26'
    )
    relu26 = tf.nn.relu(tf.layers.batch_normalization(conv26,training=True))
    # b26 = int_b(shape=[out26_channel],name='b26')
    # relu26 = tf.nn.relu(tf.nn.bias_add(conv26, b26))

    # conv3
    w27 = int_w(shape=[conv27_size, conv27_size, inp27_channel, out27_channel], name='W27')
    conv27 = tf.nn.conv2d(
        relu26, w27, strides=[1, 1, 1, 1], padding='SAME', name='conv27'
    )
    relu27 = tf.nn.relu(tf.layers.batch_normalization(conv27,training=True))

    # b27 = int_b(shape=[out27_channel], name='b27')
    # relu27 = tf.nn.relu(tf.nn.bias_add(conv27, b27))
    relu27 = tf.nn.dropout(relu27, keep_prob=dropout_value)  # dropout

    # conv4
    w28 = int_w(shape=[conv28_size, conv28_size, inp28_channel, out28_channel], name='W28')
    conv28 = tf.nn.conv2d(
        relu27, w28, strides=[1, 1, 1, 1], padding='SAME', name='conv28'
    )
    relu28 = tf.nn.relu(tf.layers.batch_normalization(conv28,training=True))

    # b28 = int_b(shape=[out28_channel], name='b28')
    # relu28 = tf.nn.relu(tf.nn.bias_add(conv28, b28))
    # conv5
    w29 = int_w(shape=[conv29_size, conv29_size, inp29_channel, out29_channel], name='W29')
    conv29 = tf.nn.conv2d(
        relu28, w29, strides=[1, 1, 1, 1], padding='SAME', name='conv29'
    )
    b29 = int_b(shape=[out29_channel], name='b29')
    # relu29 = tf.nn.relu(tf.nn.bias_add(conv29, b29))
    # # conv6
    # w30 = int_w(shape=[conv30_size, conv30_size, inp30_channel, out30_channel], name='W30')
    # conv30 = tf.nn.conv2d(
    #     relu29, w30, strides=[1, 1, 1, 1], padding='SAME', name='conv30'
    # )
    # b30 = int_b(shape=[out30_channel], name='b30')
    # # relu30 = tf.nn.relu(tf.nn.bias_add(conv30, b30))
    net_out = tf.nn.bias_add(conv29, b29)

    return net_out

