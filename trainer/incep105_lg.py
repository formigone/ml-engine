import tensorflow as tf

import runner
from util import inception_block, flatten
from graph_utils import log_conv_kernel


def branch_incep(x, name, mode, params):
  with tf.variable_scope(name):
    conv1 = tf.layers.conv2d(x, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    conv1b = tf.layers.conv2d(conv1, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1b')
    pool1 = tf.layers.max_pooling2d(conv1b, pool_size=[2, 2], strides=2, name='pool1')
    if params['verbose_summary']:
      log_conv_kernel('{}/conv1'.format(name))
      log_conv_kernel('{}/conv1b'.format(name))
      tf.summary.image('pool1', pool1[:, :, :, 0:1])

    incep2 = inception_block(pool1, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep2')

    conv3 = tf.layers.conv2d(incep2, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
    conv3b = tf.layers.conv2d(conv3, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv3b')
    pool3 = tf.layers.max_pooling2d(conv3b, pool_size=[2, 2], strides=2, name='pool3')
    if params['verbose_summary']:
      log_conv_kernel('{}/conv3'.format(name))
      log_conv_kernel('{}/conv3b'.format(name))
      tf.summary.image('pool3', pool3[:, :, :, 0:1])

    conv5 = tf.layers.conv2d(pool3, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5')
    conv5b = tf.layers.conv2d(conv5, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv5b')
    pool5 = tf.layers.max_pooling2d(conv5b, pool_size=[2, 2], strides=2, name='pool5')
    if params['verbose_summary']:
      log_conv_kernel('{}/conv5'.format(name))
      log_conv_kernel('{}/conv5b'.format(name))
      tf.summary.image('pool5', pool5[:, :, :, 0:1])

    incep6 = inception_block(pool5, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep6')

    conv7 = tf.layers.conv2d(incep6, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7')
    conv7b = tf.layers.conv2d(conv7, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv7b')
    pool7 = tf.layers.max_pooling2d(conv7b, pool_size=[2, 2], strides=2, name='pool7')
    if params['verbose_summary']:
      log_conv_kernel('{}/conv7'.format(name))
      log_conv_kernel('{}/conv7b'.format(name))
      tf.summary.image('pool7', pool7[:, :, :, 0:1])

    conv8 = tf.layers.conv2d(pool7, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv8')
    conv8b = tf.layers.conv2d(conv8, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv8b')
    pool8 = tf.layers.max_pooling2d(conv8b, pool_size=[2, 2], strides=2, name='pool8')
    if params['verbose_summary']:
      log_conv_kernel('{}/conv8'.format(name))
      log_conv_kernel('{}/conv8b'.format(name))
      tf.summary.image('pool8', pool8[:, :, :, 0:1])

    return pool8


def branch_deep(x, name, mode, params):
  with tf.variable_scope(name):
    training = mode == tf.estimator.ModeKeys.TRAIN
    norm = x
    for units in [12000, 8000, 4000, 2000]:
      with tf.variable_scope('{}_{}'.format(name, units)):
        dropout = tf.layers.dropout(norm, rate=params['dropout_rate'], training=training, name='dropout')
        dense = tf.layers.dense(dropout, units=units, activation=tf.nn.relu, name='dense')
        norm = tf.layers.batch_normalization(dense, training=mode == tf.estimator.ModeKeys.TRAIN, name='norm_merge')
    return norm


def model_fn(features, labels, mode, params):
  with tf.variable_scope('x_prep'):
    x = tf.reshape(features, [-1, 125, 161, 2], name='input_incep105_lg')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    x_spec = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')
    x_freq = tf.reshape(x_norm[:, :, :, 1], [-1, 125, 161], name='reshape_freq')
    x_freq = tf.slice(x_freq, [0, 0, 0], [-1, 125, 128], name='slice_freq')
    x_freq = tf.reshape(x_freq, [-1, 125 * 128], name='flatten_freq')

  if params['verbose_summary']:
    with tf.variable_scope('input'):
      tf.summary.audio('input_freq', x_freq, 16000)
      for sample in range(3):
        tf.summary.image('input_spec_{}'.format(sample), x_spec)

  branch_spec = branch_incep(x_spec, 'branch_spec', mode, params)
  branch_freq = branch_deep(x_freq, 'branch_freq', mode, params)

  with tf.variable_scope('branch_concat'):
    flat_spec = flatten(branch_spec, 'flat_spec')
    branch_merge = tf.concat([flat_spec, branch_freq], axis=1, name='branch_merge')
  norm = tf.layers.batch_normalization(branch_merge, training=mode == tf.estimator.ModeKeys.TRAIN, name='norm_merge')
  dense = tf.layers.dense(norm, units=2048, activation=tf.nn.relu, name='dense')
  dropout = tf.layers.dropout(dense, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')

  logits = tf.layers.dense(dropout, units=params['num_classes'], name='logits')

  predictions = {
    'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
    'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['num_classes'], name='onehot_labels')
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
  tf.summary.scalar('loss', loss)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
  }

  tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops
  )


if __name__ == '__main__':
  runner.run(model_fn)
