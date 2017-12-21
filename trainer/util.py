import tensorflow as tf

from graph_utils import log_conv_kernel


def flatten(input, name='flatten'):
  dim = input.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  return tf.reshape(input, [-1, dim], name=name)


def touch_incep(input, name):
  dim = input.get_shape()[1:]
  return tf.reshape(input, [-1, dim[0], dim[1], dim[2]], name=name)


def double_inception(prev, block_depth, name):
  tower = inception_block(prev, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth, name=name)
  tower_res = touch_incep(tower, '{}_res'.format(name))
  return inception_block(tower_res, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth, name='{}_2'.format(name))


def conv_group(prev, filters, name='conv_group', verbose=False):
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(prev, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv_same')
    convb = tf.layers.conv2d(conv, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convb')
    convc = tf.layers.conv2d(convb, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convc')
    pool = tf.layers.max_pooling2d(convc, pool_size=[2, 2], strides=2, name='pool')
    if verbose:
      log_conv_kernel('{}/conv_same'.format(name))
      log_conv_kernel('{}/convb'.format(name))
      log_conv_kernel('{}/convc'.format(name))
      tf.summary.image('{}/pool'.format(name), pool[:, :, :, 0:1])
    return pool


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
