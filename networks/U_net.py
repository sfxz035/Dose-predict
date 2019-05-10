from networks.ops import *
import tensorflow as tf
import numpy as np
#paramaters
FILTER_DIM = 24
OUTPUT_C = 1
#deep 5
def inference(images,is_training=True,reuse = False,dropout_value=0.25,name='UNet'):
    with tf.variable_scope(name, reuse=reuse):
        L1_1 = ReLU(conv_bn(images, FILTER_DIM, k_h=3,is_train=is_training, name='Conv2d_1_1'),name='ReLU_1_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM / (FILTER_DIM*64), 1 / 4)
        L1_1 = tf.nn.dropout(L1_1, keep_prob=dropout_value1)  # dropout
        L1_2 = ReLU(conv_bn(L1_1, FILTER_DIM, k_h=3,is_train=is_training, name='Conv2d_1_2'),name='ReLU_1_2')
        L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling1')  ##


        L2_2 = ReLU(conv_bn(L2_1, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_1'),name='ReLU_2_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*2 / (FILTER_DIM*64), 1 / 4)
        L2_2 = tf.nn.dropout(L2_2, keep_prob=dropout_value1)  # dropout
        L2_3 = ReLU(conv_bn(L2_2, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_2'),name='ReLU_2_2')
        L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling2')    ##

        L3_2 = ReLU(conv_bn(L3_1, FILTER_DIM*4, k_h=3, is_train=is_training,name='Conv2d_3_1'),name='ReLU_3_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*4 / (FILTER_DIM*64), 1 / 4)
        L3_2 = tf.nn.dropout(L3_2, keep_prob=dropout_value1)  # dropout
        L3_3 = ReLU(conv_bn(L3_2, FILTER_DIM*4, k_h=3, is_train=is_training,name='Conv2d_3_2'),name='ReLU_3_2')
        L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling3')    ##

        L4_2 = ReLU(conv_bn(L4_1, FILTER_DIM*8, k_h=3, is_train=is_training,name='Conv2d_4_1'),name='ReLU_4_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*8 / (FILTER_DIM*64), 1 / 4)
        L4_2 = tf.nn.dropout(L4_2, keep_prob=dropout_value1)  # dropout
        L4_3 = ReLU(conv_bn(L4_2, FILTER_DIM*8, k_h=3, is_train=is_training,name='Conv2d_4_2'),name='ReLU_4_2')
        L5_1 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L5_2 = ReLU(conv_bn(L5_1, FILTER_DIM*16, k_h=3, is_train=is_training,name='Conv2d_5_1'),name='ReLU_5_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*16 / (FILTER_DIM*64), 1 / 4)
        L5_2 = tf.nn.dropout(L5_2, keep_prob=dropout_value1)  # dropout
        L5_3 = ReLU(conv_bn(L5_2, FILTER_DIM*16, k_h=3, is_train=is_training,name='Conv2d_5_2'),name='ReLU_5_2')
        L6_1 = tf.nn.max_pool(L5_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L6_2 = ReLU(conv_bn(L6_1, FILTER_DIM*32, k_h=3, is_train=is_training,name='Conv2d_6_1'),name='ReLU_6_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*32 / (FILTER_DIM*64), 1 / 4)
        L6_2 = tf.nn.dropout(L6_2, keep_prob=dropout_value1)  # dropout
        L6_3 = ReLU(conv_bn(L6_2, FILTER_DIM*32, k_h=3, is_train=is_training,name='Conv2d_6_2'),name='ReLU_6_2')
        L7_1 = tf.nn.max_pool(L6_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME',name = 'MaxPooling4')  ##

        L7_2 = ReLU(conv_bn(L7_1, FILTER_DIM*64, k_h=3, is_train=is_training,name='Conv2d_7_1'),name='ReLU_7_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*64 / (FILTER_DIM*64), 1 / 4)
        L7_2 = tf.nn.dropout(L7_2, keep_prob=dropout_value1)  # dropout
        L7_3 = ReLU(conv_bn(L7_2, FILTER_DIM*64, k_h=3, is_train=is_training,name='Conv2d_7_2'),name='ReLU_7_2')

        L6_U1 = ReLU(Deconv2d_bn(L7_3, L6_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d6'),name='DeReLU6')
        L6_U1 = tf.concat((L6_3, L6_U1), -1)
        L6_U2 = ReLU(conv_bn(L6_U1, FILTER_DIM * 32, k_h=3, is_train=is_training,name='Conv2d_6_u1'),name='ReLU_6_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*32 / (FILTER_DIM*64), 1 / 4)
        L6_U2 = tf.nn.dropout(L6_U2, keep_prob=dropout_value1)  # dropout
        L6_U3 = ReLU(conv_bn(L6_U2, FILTER_DIM * 32, k_h=3, is_train=is_training,name='Conv2d_6_u2'),name='ReLU_6_u2')

        L5_U1 = ReLU(Deconv2d_bn(L6_U3, L5_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d5'),name='DeReLU5')
        L5_U1 = tf.concat((L5_3, L5_U1), -1)
        L5_U2 = ReLU(conv_bn(L5_U1, FILTER_DIM * 16, k_h=3, is_train=is_training,name='Conv2d_5_u1'),name='ReLU_5_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*16 /(FILTER_DIM*64), 1 / 4)
        L5_U2 = tf.nn.dropout(L5_U2, keep_prob=dropout_value1)  # dropout
        L5_U3 = ReLU(conv_bn(L5_U2, FILTER_DIM * 16, k_h=3, is_train=is_training,name='Conv2d_5_u2'),name='ReLU_5_u2')

        L4_U1 = ReLU(Deconv2d_bn(L5_U3, L4_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d4'),name='DeReLU4')
        L4_U1 = tf.concat((L4_3, L4_U1), -1)
        L4_U2 = ReLU(conv_bn(L4_U1, FILTER_DIM * 8, k_h=3, is_train=is_training,name='Conv2d_4_u1'),name='ReLU_4_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*8 / (FILTER_DIM*64), 1 / 4)
        L4_U2 = tf.nn.dropout(L4_U2, keep_prob=dropout_value1)  # dropout
        L4_U3 = ReLU(conv_bn(L4_U2, FILTER_DIM * 8, k_h=3, is_train=is_training,name='Conv2d_4_u2'),name='ReLU_4_u2')

        L3_U1 = ReLU(Deconv2d_bn(L4_U3,L3_3.get_shape(),k_h = 3,is_train=is_training,name = 'Deconv2d3'),name = 'DeReLU3')
        L3_U1 = tf.concat((L3_3, L3_U1), -1)
        L3_U2 = ReLU(conv_bn(L3_U1, FILTER_DIM*4, k_h=3,is_train=is_training, name='Conv2d_3_u1'), name='ReLU_3_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*4 / (FILTER_DIM*64), 1 / 4)
        L3_U2 = tf.nn.dropout(L3_U2, keep_prob=dropout_value1)  # dropout
        L3_U3 = ReLU(conv_bn(L3_U2, FILTER_DIM*4, k_h=3,is_train=is_training, name='Conv2d_3_u2'), name='ReLU_3_u2')

        L2_U1 = ReLU(Deconv2d_bn(L3_U3,L2_3.get_shape(), k_h = 3,is_train=is_training,name = 'Deconv2d2'),name='DeReLU2')
        L2_U1 = tf.concat((L2_3, L2_U1), -1)
        L2_U2 = ReLU(conv_bn(L2_U1, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_u1'),name='ReLU_2_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*2 / (FILTER_DIM*64), 1 / 4)
        L2_U2 = tf.nn.dropout(L2_U2, keep_prob=dropout_value1)  # dropout
        L2_U3 = ReLU(conv_bn(L2_U2, FILTER_DIM*2, k_h=3, is_train=is_training,name='Conv2d_2_u2'),name='ReLU_2_u2')

        L1_U1 = ReLU(Deconv2d_bn(L2_U3, L1_2.get_shape(),k_h=3,is_train=is_training,name='Deconv2d1'),name='DeReLU1')
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = ReLU(conv_bn(L1_U1, FILTER_DIM, k_h=3, is_train=is_training,name='Conv2d_1_u1'),name='ReLU_1_u1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM*2 / (FILTER_DIM*64), 1 / 4)
        L1_U2 = tf.nn.dropout(L1_U2, keep_prob=dropout_value1)  # dropout
        L1_U3 = ReLU(conv_bn(L1_U2, FILTER_DIM, k_h=3, is_train=is_training,name='Conv2d_1_u2'),name='ReLU_1_u2')

        Lout_1 = conv_relu(L1_U3, FILTER_DIM, k_h=3,name='Conv2d_Lout_1')
        dropout_value1 = 1 - dropout_value * pow(FILTER_DIM / (FILTER_DIM*64), 1 / 4)
        Lout_1 = tf.nn.dropout(Lout_1, keep_prob=dropout_value1)  # dropout
        Lout_2 = conv_relu(Lout_1, FILTER_DIM/2, k_h=3,name='Conv2d_Lout_2')
        Lout_3 = conv_relu(Lout_2, FILTER_DIM/4, k_h=3,name='Conv2d_Lout_3')
        dropout_value1 = 1 - dropout_value * pow((FILTER_DIM/4) / (FILTER_DIM*64), 1 / 4)
        Lout_3 = tf.nn.dropout(Lout_3, keep_prob=dropout_value1)  # dropout
        Lout_4 = conv_relu(Lout_3, 3, k_h=3, name='Conv2d_Lout_4')
        Lout_5 = conv_relu(Lout_4, 2, k_h=3,name='Conv2d_Lout_5')
        dropout_value1 = 1 - dropout_value * pow(2 / (FILTER_DIM*64), 1 / 4)
        Lout_5 = tf.nn.dropout(Lout_5, keep_prob=dropout_value1)  # dropout
        Lout_6 = conv_relu(Lout_5, 1, k_h=3,name='Conv2d_Lout_6')

        out = conv(Lout_6, OUTPUT_C,name='Conv1d_out')

    # variables = tf.contrib.framework.get_variables(name)

    return out









