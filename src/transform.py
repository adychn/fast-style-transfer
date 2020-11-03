import tensorflow as tf, pdb
import tensorflow.compat.v1 as v1
import sys

WEIGHTS_INIT_STDEV = .1

def net(image):
    conv1 = _conv_layer(image, 32, filter_size=9, strides=1) # same
    conv2 = _conv_layer(conv1, 64, filter_size=3, strides=2) # divide by 2
    conv3 = _conv_layer(conv2, 128, filter_size=3, strides=2) # divide by 2
    resid1 = _residual_block(conv3, filter_size=3) # same
    resid2 = _residual_block(resid1, 3) # same
    resid3 = _residual_block(resid2, 3) # same
    resid4 = _residual_block(resid3, 3) # same
    resid5 = _residual_block(resid4, 3) # same
    conv_t1 = _conv_tranpose_layer(resid5, 64, filter_size=3, strides=2) # multiply by 2
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, filter_size=3, strides=2) # multiply by 2
    conv_t3 = _conv_layer(conv_t2, 3, filter_size=9, strides=1, relu=False) # same
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

# padding with SAME plus stride of 1 will keep the same size output image.
def _conv_layer(net, out_channels, filter_size, strides, relu=True):
    filters_init = _conv_init_vars(net, out_channels, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(input=net, filters=filters_init, strides=strides_shape, padding='SAME')
    # net = _instance_norm(net)
    net = _batch_instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

# SAME: output_size = input_size * stride
# VALID: output_size = (input_size - 1) * stride + filter_size
def _conv_tranpose_layer(net, out_channels, filter_size, strides, debug=False):
    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]

    # Better Upsampling, https://distill.pub/2016/deconv-checkerboard/
    new_height = int(rows * strides)
    new_width = int(cols * strides)
    if debug:
        print(f"Before resize: {net}")
    net = tf.image.resize(net, size=[new_height, new_width], method='nearest')
    if debug:
        print(f"After resize: {net}")
    # hard to pad with various image sizes from the input.
    # paddings = tf.constant([[0, 0], [x, x], [x, x], [0, 0]])
    # net = tf.pad(net, paddings, "REFLECT")
    filters_init = _conv_init_vars(net, out_channels, filter_size)
    net = tf.nn.conv2d(net, filters=filters_init, strides=1, padding='SAME')

    # previously
    # strides_shape = [1, strides, strides, 1]
    # new_shape = [batch_size, int(rows * strides), int(cols * strides), out_channels]
    # tf_shape = tf.stack(new_shape)
    # net = tf.nn.conv2d_transpose(input=net, filters=filters_init, output_shape=tf_shape,
    #                              strides=strides_shape, padding='SAME')
    # net = _instance_norm(net)
    net = _batch_instance_norm(net)
    net = tf.nn.relu(net)
    return net

def _conv_init_vars(net, out_channels, filter_size):   #, transpose=False):
    _, rows, cols, in_channels = [i for i in net.get_shape()]
    weights_shape = [filter_size, filter_size, in_channels, out_channels]
    # previously
    # if not transpose:
    #     # filters
    #     # A Tensor. Must have the same type as input.
    #     # A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    #     weights_shape = [filter_size, filter_size, in_channels, out_channels]
    # else:
    #     weights_shape = [filter_size, filter_size, out_channels, in_channels]
    normal_init = tf.random.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
    filters_init = tf.Variable(normal_init, dtype=tf.float32)
    return filters_init

# padding with SAME plus stride of 1 will keep the same size output image.
def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, strides=1)
    tmp = _conv_layer(tmp, 128, filter_size, strides=1)
    return net + tmp

def _instance_norm(x):
    N, H, W, C = [i for i in x.get_shape()]
    eps = 1e-5

    # normalzied, (N, H, W, C)
    mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    # per-channel learnable linear transform
    gamma = tf.Variable(tf.ones([C]))
    beta = tf.Variable(tf.zeros([C]))
    x_hat = x * gamma + beta

    return x_hat

def _batch_instance_norm(x, train=True):
    N, H, W, C = [i for i in x.get_shape()]
    eps = 1e-5

    # batch normalized, for better detection of the pre-transformed object
    batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2], keepdims=True)
    x_batch = (x - batch_mean) / tf.sqrt(batch_var + eps)

    # instance normalizaed, for style manipulation
    ins_mean, ins_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    x_ins = (x - ins_mean) / tf.sqrt(ins_var + eps)

    rho = tf.Variable(tf.ones([C]),
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
    gamma = tf.Variable(tf.ones([C]))
    beta = tf.Variable(tf.zeros([C]))

    x_hat = rho * x_batch + (1 - rho) * x_ins
    x_hat = x_hat * gamma + beta

    return x_hat
