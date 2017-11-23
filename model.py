import tensorflow as tf
from resnet import softmax_layer, conv_layer, residual_block

def resnet(inpt, num_conv):
    layers = []

    with tf.name_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.name_scope('conv2_%d' % (i + 1)):
            conv2_x = residual_block(layer2[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

            assert conv2.get_shape().as_list()[1:] == [32, 32, 16]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, down_sample)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

            assert conv3.get_shape().as_list()[1:] == [16, 16, 32]

    for i in range(num_conv):
        down_sample = True if i == 0 else False
        with tf.name_scope('conv4_%d' % (i + 1)):
            conv4_x = residual_block(layers[-1], 32, down_sample)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv4_x)
            layers.append(conv4)

            assert conv4.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.name_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])

        assert global_pool.get_shape().as_list()[1:] == [64]

        out = softmax_layer(global_pool, [64, 10])
        layers.append(out)

    return layers[-1]

