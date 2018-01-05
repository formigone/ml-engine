import tensorflow as tf

import runner
from util import inception_block, flatten
from graph_utils import log_conv_kernel


def branch(x, name, params):
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

    flat = flatten(pool8)
    return flat


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 125, 161, 2], name='input_incep102_lg')
  x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
  x_a = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1])
  x_b = tf.reshape(x_norm[:, :, :, 1], [-1, 125, 161, 1])

  if params['verbose_summary']:
    with tf.variable_scope('input'):
      for sample in range(3):
        tf.summary.image('input_spec_{}'.format(sample), x_a)
        tf.summary.image('input_freq_{}'.format(sample), x_b)

  flat_a = branch(x_a, 'branch_a', params)
  dense_a = tf.layers.dense(flat_a, units=2048, activation=tf.nn.relu, name='dense_a')
  dropout_a = tf.layers.dropout(dense_a, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN,
                                name='dropout_a')

  flat_b = branch(x_b, 'branch_b', params)
  dense_b = tf.layers.dense(flat_b, units=2048, activation=tf.nn.relu, name='dense_b')
  dropout_b = tf.layers.dropout(dense_b, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN,
                                name='dropout_b')

  dense_merge = tf.concat([dropout_a, dropout_b], 1, name='dense_merge')

  logits = tf.layers.dense(dense_merge, units=params['num_classes'], name='logits')

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
