import tensorflow as tf
import numpy as np
from math import ceil

VGG_MEAN = [103.939, 116.779, 123.68]
#VGG_MEAN = [0, 0, 0]

def weight_variable(shape, name = None):
    initial = tf.truncated_normal(shape, stddev = 0.02)
    return tf.Variable(initial, name = name)

def bias_variable(shape, name = None):
    #initial = tf.Variable(tf.zeros([shape]))
    initial = tf.truncated_normal([shape], stddev = 0.005)
    return tf.Variable(initial, name = name)

def batch_norm_layer(x, train_phase, scope_bn, name = 'batch_norm'):
    shape = x.get_shape().as_list()
    x_unrolled = tf.reshape(x, [-1, shape[1] * shape[2] * shape[3]])
    with tf.variable_scope(scope_bn):
        beta = tf.Variable(tf.constant(0.0, shape = [x_unrolled.shape[-1]]), name = 'beta', trainable = True)
        gamma = tf.Variable(tf.constant(1.0, shape = [x_unrolled.shape[-1]]), name = 'gamma', trainable = True)
        batch_mean, batch_var = tf.nn.moments(x_unrolled, axes = [0], name = 'moments')
        ema = tf.train.ExponentialMovingAverage(decay = 0.5)
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(train_phase, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x_unrolled, mean, var, beta, gamma, 1e-3)
        normed = tf.reshape(normed, tf.shape(x))
    return normed

def conv_layer(input, filter_shape, stride = 1, batch = True, name = 'conv'):
    out_channels = filter_shape[3]
    filter_ = weight_variable(filter_shape)
    bias_ = bias_variable(out_channels)
    conv = tf.nn.conv2d(input, filter = filter_, strides = [1, stride, stride, 1], padding = 'SAME')
    mean, var = tf.nn.moments(conv, [0, 1, 2])
    beta = tf.Variable(tf.zeros([out_channels]))
    gamma = weight_variable([out_channels])
    if batch is True:
        conv = tf.nn.batch_norm_with_global_normalization(conv, mean, var, beta, gamma, 0.0001, scale_after_normalization = True)
    out = tf.nn.bias_add(conv, bias_)
    out = tf.nn.relu(tf.nn.bias_add(conv, bias_))
    return out

def max_pool_layer(input, size, stride, name = 'maxpool'):
    return tf.nn.max_pool(input, ksize = [1, size, size, 1], strides = [1, stride, stride, 1], padding = 'SAME')

def residual_block(input, output_depth, down_sample = False, is_training = True, name = 'residual'):
    input_depth = input.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 3, 3, 1]
        input = tf.nn.max_pool(input, ksize = filter_, strides = filter_, padding = 'SAME')
    conv1 = conv_layer(input, [1, 1, input_depth, output_depth])
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth])
    del conv1
    conv3 = conv_layer(conv2, [1, 1, output_depth, output_depth * 4])
    del conv2
    if input_depth != output_depth * 4:
        input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth * 4 - input_depth]])
    else:
        input_layer = input
    del input
    res = conv3 + input_layer
    del conv3, input_layer
    if is_training is True:
        res = tf.nn.dropout(res, keep_prob = 0.8)
    return res

def residual_block_v2(input, output_depth, down_sample = False, is_training = True, name = 'residual'):
    input_depth = input.get_shape().as_list()[3]
    if down_sample:
        filter_ = [1, 3, 3, 1]
        input = tf.nn.max_pool(input, ksize = filter_, strides = filter_, padding = 'SAME')
    conv1 = conv_layer(input, [3, 3, input_depth, output_depth])
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth * 2])
    del conv1
    if input_depth != output_depth * 2:
        input_layer = tf.pad(input, [[0, 0], [0, 0], [0, 0], [0, output_depth * 2 - input_depth]])
    else:
        input_layer = input
    del input
    res = conv2 + input_layer
    del conv2, input_layer
    if is_training is True:
        res = tf.nn.dropout(res, keep_prob = 0.5)
    return res

def unpooling_layer(input, name = 'unpooling'):
    with tf.name_scope(name) as scope:
        shape = input.get_shape().as_list()
        dim = len(shape[1:-1])
        out = (tf.reshape(input, [-1] + shape[-dim:]))
        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)
        out_size = [-1] + [s * 2 for s in shape[1: -1]] + [shape[-1]]
        out = tf.reshape(out, out_size, name = scope)
    return out

def up_proj(input, kernel_0, kernel, stride = 1, name = 'up_projection'):
    max_pool_0 = unpooling_layer(input)
    output_0 = conv_layer(max_pool_0, kernel_0, stride)
    output_1 = conv_layer(output_0, kernel[1], stride)
    return tf.nn.relu(output_1 + output_0)
    #conv_x = conv_layer(max_pool_0, kernel[0], stride)
    #conv_x = tf.nn.relu(conv_x)
    #conv_x = conv_layer(conv_x, kernel[1], stride)
    #return tf.nn.relu(conv_x + output_0)

def construct_layer(input, is_training = True):
    red, green, blue, depth = tf.split(input, 4, 3)
    input = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2], depth], 3)
    #input = input / 255
    
    with tf.variable_scope('conv1'):
        conv1 = conv_layer(input, [7, 7, 4, 64])
    conv2 = max_pool_layer(conv1, 3, 2)
    """
    with tf.variable_scope('conv2'):
        for i in range(3):
            conv2 = residual_block(conv2, 64, False, is_training)    
    conv3 = max_pool_layer(conv2, 3, 2)
    with tf.variable_scope('conv3'):
        for i in range(4):
            conv3 = residual_block(conv3, 128, False, is_training)
    conv4 = max_pool_layer(conv3, 3, 2)
    with tf.variable_scope('conv4'):
        for i in range(6):
            conv4 = residual_block(conv4, 256, False, is_training)
    conv5 = max_pool_layer(conv4, 3, 2)
    with tf.variable_scope('conv5'):
        for i in range(3):
            conv5 = residual_block(conv5, 512, False, is_training)
    conv6 = max_pool_layer(conv5, 3, 2)
    """
    with tf.variable_scope('conv2'):
        for i in range(2):
            conv2 = residual_block_v2(conv2, 64, False, is_training)    
    conv3 = max_pool_layer(conv2, 3, 2)
    del conv2
    with tf.variable_scope('conv3'):
        for i in range(2):
            conv3 = residual_block_v2(conv3, 128, False, is_training)
    conv4 = max_pool_layer(conv3, 3, 2)
    del conv3
    with tf.variable_scope('conv4'):
        for i in range(2):
            conv4 = residual_block_v2(conv4, 256, False, is_training)
    conv5 = max_pool_layer(conv4, 3, 2)
    del conv4
    with tf.variable_scope('conv5'):
        for i in range(2):
            conv5 = residual_block_v2(conv5, 512, False, is_training)
    conv6 = max_pool_layer(conv5, 3, 2)
    del conv5
    #print(conv6)
    with tf.variable_scope('conv6'):
        channels = conv6.get_shape().as_list()[3]
        conv6 = conv_layer(conv6, [1, 1, channels, channels // 2])
    up_proj_x = conv6
    del conv6
    #epoch_size = 512
    epoch_size = up_proj_x.get_shape().as_list()[-1] // 2
    for i in range(4):
        with tf.variable_scope('deconv{}'.format(i + 1)):
            kernel_0 = [5, 5, epoch_size * 2, epoch_size]
            kernel = []
            kernel.append([5, 5, epoch_size * 2, epoch_size])
            kernel.append([3, 3, epoch_size, epoch_size])
            epoch_size = epoch_size // 2
            up_proj_x = up_proj(up_proj_x, kernel_0, kernel, 1)
    decode_1 = up_proj_x
    del up_proj_x
    #conv7 = conv_layer(decode_1, [3, 3, 64, 1])
    conv7 = conv_layer(decode_1, [3, 3, 32, 1])
    del decode_1
    result = tf.image.resize_bilinear(conv7, tf.Variable([2 * conv7.get_shape().as_list()[1], 2 * conv7.get_shape().as_list()[2]]))
    #prediction = tf.argmax(result, dimension = 3, name = 'prediction')
    return result
        
        
        
        
        
