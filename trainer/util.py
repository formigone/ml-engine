import tensorflow as tf


def flatten(input, name='flatten'):
  dim = input.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  return tf.reshape(input, [-1, dim], name=name)


def inception_block(prev, t1x1=2, t3x3=2, t5x5=2, tmp=2, name='incep', log_conv_weights=False):
  with tf.variable_scope(name):
    with tf.variable_scope('1x1_conv'):
      tower_1x1 = tf.layers.conv2d(prev,
                                   filters=t1x1,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')

    with tf.variable_scope('3x3_conv'):
      tower_3x3 = tf.layers.conv2d(prev,
                                   filters=t3x3,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      tower_3x3 = tf.layers.conv2d(tower_3x3,
                                   filters=t3x3,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv')

    with tf.variable_scope('5x5_conv'):
      tower_5x5 = tf.layers.conv2d(prev,
                                   filters=t5x5,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      tower_5x5 = tf.layers.conv2d(tower_5x5,
                                   filters=t5x5,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv_1')
      tower_5x5 = tf.layers.conv2d(tower_5x5,
                                   filters=t5x5,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv_2')

    with tf.variable_scope('maxpool'):
      tower_mp = tf.layers.max_pooling2d(prev,
                                         pool_size=3,
                                         strides=1,
                                         padding='same',
                                         name='3x3_maxpool')
      tower_mp = tf.layers.conv2d(tower_mp,
                                  filters=tmp,
                                  kernel_size=1,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='1x1_conv')
    return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)
