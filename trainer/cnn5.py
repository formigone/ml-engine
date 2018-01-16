import tensorflow as tf

from util import conf_mat, flatten
from graph_utils import log_conv_kernel
import runner

ModeKeys = tf.estimator.ModeKeys
EstimatorSpec = tf.estimator.EstimatorSpec


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 125, 161, 2], name='cnn5')
  x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
  x = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')
  training = mode == ModeKeys.TRAIN

  if params['verbose_summary']:
    tf.summary.image('input', x)


  conv = x
  conv = tf.layers.conv2d(conv, filters=16, kernel_size=5, activation=tf.nn.relu, name='conv1')
  conv = tf.layers.conv2d(conv, filters=32, kernel_size=5, activation=tf.nn.relu, name='conv2')
  dropout = tf.layers.dropout(conv, rate=0.1, training=training, name='dropout1')
  conv = dropout
  conv = tf.layers.conv2d(conv, filters=64, kernel_size=5, activation=tf.nn.relu, name='conv3')
  conv = tf.layers.conv2d(conv, filters=128, kernel_size=5, activation=tf.nn.relu, name='conv4')
  dropout = tf.layers.dropout(conv, rate=0.1, training=training, name='dropout1')
  conv = dropout
  pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, name='pool4')
  if params['verbose_summary']:
    log_conv_kernel('conv1')
    log_conv_kernel('conv2')
    log_conv_kernel('conv3')
    log_conv_kernel('conv4')
    tf.summary.image('pool4', pool[:, :, :, 0:1])

  conv = pool
  conv = tf.layers.conv2d(conv, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv5')
  dropout = tf.layers.dropout(conv, rate=0.1, training=training, name='dropout1')
  conv = dropout
  conv = tf.layers.conv2d(conv, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv6')
  conv = tf.layers.conv2d(conv, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv7')
  dropout = tf.layers.dropout(conv, rate=0.1, training=training, name='dropout1')
  conv = dropout
  conv = tf.layers.conv2d(conv, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv8')
  pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, name='pool8')
  if params['verbose_summary']:
    log_conv_kernel('conv5')
    log_conv_kernel('conv6')
    log_conv_kernel('conv7')
    log_conv_kernel('conv8')
    tf.summary.image('pool8', pool[:, :, :, 0:1])

  flat = flatten(pool)

  dropout = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == ModeKeys.TRAIN, name='dropout')
  dense = tf.layers.dense(dropout, units=1024, activation=tf.nn.relu, name='dense')

  logits = tf.layers.dense(dense, units=params['num_classes'], name='logits')

  predictions = {
    'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
    'probabilities': tf.nn.softmax(logits, name='prediction_probabilities'),
  }

  if mode == ModeKeys.PREDICT:
    return EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

  tf.summary.image('confusion_matrix', conf_mat(labels, predictions['classes'], params['num_classes']))

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


if __name__ == '__main__':
    runner.run(model_fn)
