import tensorflow as tf

from graph_utils import log_conv_kernel

ModeKeys = tf.estimator.ModeKeys
EstimatorSpec = tf.estimator.EstimatorSpec


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 99, 161, 1], name='input_cnn3')
  x_norm = tf.layers.batch_normalization(x, training=mode == ModeKeys.TRAIN, name='x_norm')
  if params['verbose_summary']:
    tf.summary.image('input', x)

  conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
  conv1b = tf.layers.conv2d(conv1, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1b')
  pool1 = tf.layers.max_pooling2d(conv1b, pool_size=[2, 2], strides=2, name='pool1')
  if params['verbose_summary']:
    log_conv_kernel('conv1')
    tf.summary.image('pool1', pool1[:, :, :, 0:1])

  conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
  conv2b = tf.layers.conv2d(conv2, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2b')
  pool2 = tf.layers.max_pooling2d(conv2b, pool_size=[2, 2], strides=2, name='pool2')
  if params['verbose_summary']:
    log_conv_kernel('conv2')
    tf.summary.image('pool2', pool2[:, :, :, 0:1])

  conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
  conv3b = tf.layers.conv2d(conv3, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3b')
  pool3 = tf.layers.max_pooling2d(conv3b, pool_size=[2, 2], strides=2, name='pool3')
  if params['verbose_summary']:
    log_conv_kernel('conv3')
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

  conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv4')
  conv4b = tf.layers.conv2d(conv4, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv4b')
  pool4 = tf.layers.max_pooling2d(conv4b, pool_size=[2, 2], strides=2, name='pool4')
  if params['verbose_summary']:
    log_conv_kernel('conv4')
    tf.summary.image('pool4', pool4[:, :, :, 0:1])

  dim = pool4.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  flat = tf.reshape(pool4, [-1, dim], name='flat')

  dropout5 = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == ModeKeys.TRAIN, name='dropout5')
  dense5 = tf.layers.dense(dropout5, units=1024, activation=tf.nn.relu, name='dense5')

  logits = tf.layers.dense(dense5, units=params['num_classes'], name='logits')

  predictions = {
    'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
    'probabilities': tf.nn.softmax(logits, name='prediction_probabilities'),
  }

  if mode == ModeKeys.PREDICT:
    return EstimatorSpec(mode=mode, predictions=predictions)

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

  return EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops
  )
