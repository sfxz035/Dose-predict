import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework import add_model_variable
from tensorflow.contrib.nccl.ops import gen_nccl_ops
from tensorflow.contrib.nccl.python.ops import nccl_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib import slim
import re
from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()

@slim.add_arg_scope
def sync_batch_norm(inputs,
                    decay=0.999,
                    epsilon=0.001,
                    activation_fn=None,
                    updates_collections=tf.GraphKeys.UPDATE_OPS,
                    is_training=True,
                    variables_collections=None,
                    trainable=True,
                    scope=None,
                    num_dev=1):
    '''
    num_dev is how many gpus you use.
    '''

    red_axises = [0, 1, 2]
    num_outputs = inputs.get_shape().as_list()[-1]

    # if scope is None:
    #     scope = inputs.name.split(':')[0].replace(tf.get_variable_scope().name, '') + '/BatchNorm'
    # print(inputs.name, tf.get_variable_scope().name, scope)
    with variable_scope.variable_scope(scope, 'BatchNorm', [inputs]):
        gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=trainable,
                                 collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                                              tf.GraphKeys.MODEL_VARIABLES,
                                              tf.GraphKeys.GLOBAL_VARIABLES])
        beta = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0), trainable=trainable,
                               collections=[tf.GraphKeys.TRAINABLE_VARIABLES,
                                            tf.GraphKeys.MODEL_VARIABLES,
                                            tf.GraphKeys.GLOBAL_VARIABLES])
        # print(gamma.name)
        moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
                                      initializer=tf.constant_initializer(0.0), trainable=False,
                                      collections=[tf.GraphKeys.MODEL_VARIABLES,
                                                   tf.GraphKeys.GLOBAL_VARIABLES])

        moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
                                     initializer=tf.constant_initializer(1.0), trainable=False,
                                     collections=[tf.GraphKeys.MODEL_VARIABLES,
                                                  tf.GraphKeys.GLOBAL_VARIABLES])

        if is_training and trainable:

            if num_dev == 1:
                mean, var = tf.nn.moments(inputs, red_axises)
            else:
                shared_name = re.sub('Model[0-9]+/', '', tf.get_variable_scope().name)

                # print('shared name', shared_name)
                batch_mean = tf.reduce_mean(inputs, axis=red_axises)
                batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)
                batch_mean = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
                batch_mean_square = gen_nccl_ops.nccl_all_reduce(
                    input=batch_mean_square,
                    reduction='sum',
                    num_devices=num_dev,
                    shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
                mean = batch_mean
                var = batch_mean_square - tf.square(batch_mean)
            outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)
            # print(outputs.device)
            if int(outputs.device[-1]) == 0:
                update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
                update_moving_var_op = tf.assign(moving_var, moving_var * decay + var * (1 - decay))
                # add_model_variable(moving_mean)
                # add_model_variable(moving_var)

                if updates_collections is None:
                    with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
                        outputs = tf.identity(outputs)
                else:
                    tf.add_to_collections(updates_collections, update_moving_mean_op)
                    tf.add_to_collections(updates_collections, update_moving_var_op)
                    outputs = tf.identity(outputs)
            else:
                outputs = tf.identity(outputs)

        else:
            outputs, _, _ = tf.nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_var,
                                                   epsilon=epsilon, is_training=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs