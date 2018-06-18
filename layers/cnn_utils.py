import tensorflow as tf


def weight_variable(w_shape):
    return tf.get_variable(name='weight', shape=w_shape, initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)


def bias_variable(b_shape):
    return tf.get_variable(name='bias', shape=b_shape, initializer=tf.constant_initializer(0.01), dtype=tf.float32)


def conv_2d(network, name, w_shape, b_shape, strides, padding='SAME', active_func=tf.nn.relu, groups=1):
    conv = lambda a, b: tf.nn.conv2d(a, b, strides=strides, padding=padding)
    new_shape = [w_shape[0], w_shape[1], w_shape[2] // groups, w_shape[3]]
    with tf.variable_scope(name):
        w = weight_variable(new_shape)
        b = bias_variable(b_shape)

        network_new = tf.split(network, num_or_size_splits=groups, axis=3)
        w_new = tf.split(w, num_or_size_splits=groups, axis=3)

        feature_map = [conv(t1, t2) for t1, t2 in zip(network_new, w_new)]
        network = tf.concat(values=feature_map, axis=3)
        # network = tf.nn.conv2d(network, w, strides=strides, padding=padding)
        network = tf.nn.bias_add(network, b)

        if active_func is not None:
            network = active_func(network)
        return network


def max_pool_2d(network, name, ksize, strides, padding='SAME'):
    with tf.variable_scope(name):
        network = tf.nn.max_pool(network, ksize=ksize, strides=strides, padding=padding)
        return network


def fully_connected(network, name, w_shape, b_shape, regularizer=None, need_dropout=False, keep_prob=1.0, active_func=tf.nn.relu):
    with tf.variable_scope(name):
        w = weight_variable(w_shape)
        b = bias_variable(b_shape)
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(w))
        network = tf.add(tf.matmul(network, w), b)

        if active_func is not None:
            network = active_func(network)

        if need_dropout:
            network = tf.nn.dropout(network, keep_prob=keep_prob)

        return network


def dropout_layer(network, keep_prob, name=None):
    return tf.nn.dropout(network, keep_prob, name)


def lrn_layer(network, depth_radius, alpha, beta, bias=1.0, name=None):
    return tf.nn.local_response_normalization(network, depth_radius, alpha, beta, bias, name)
