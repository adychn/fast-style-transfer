import tensorflow as tf, pdb
import tensorflow.compat.v1 as v1
import sys

WEIGHTS_INIT_STDEV = .1

def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(input=net, filters=weights_init, strides=strides_shape, padding='SAME')
    # net = _instance_norm(net)
    net = _batch_instance_norm(net)
    if relu:
        net = tf.nn.relu(net)
    return net

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size, rows, cols, in_channels = [i for i in net.get_shape()]

    new_shape = [batch_size, int(rows * strides), int(cols * strides), num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(input=net, filters=weights_init, output_shape=tf_shape,
                                 strides=strides_shape, padding='SAME')
    # net = _instance_norm(net)
    net = _batch_instance_norm(net)
    net = tf.nn.relu(net)
    return net

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    tmp = _conv_layer(tmp, 128, filter_size, 1)
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

    # rho = v1.get_variable("rhoo", [C], initializer=tf.constant_initializer(1.0),
    #         constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
    # gamma = v1.get_variable("gammaa", [C], initializer=tf.constant_initializer(1.0))
    # beta = v1.get_variable("betaa", [C], initializer=tf.constant_initializer(0.0))

    rho = tf.Variable(tf.ones([C]),
            constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
    gamma = tf.Variable(tf.ones([C]))
    beta = tf.Variable(tf.zeros([C]))

    x_hat = rho * x_batch + (1 - rho) * x_ins
    x_hat = x_hat * gamma + beta

    return x_hat

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]
    normal_init = tf.random.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1)
    weights_init = tf.Variable(normal_init, dtype=tf.float32)
    return weights_init

def main():
    preds = net(sys.argv[-1])
    print(preds.shape)

if __name__ == '__main__':
    main()
