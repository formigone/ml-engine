import tensorflow as tf

from util import conf_mat
from graph_utils import log_conv_kernel
import runner

ModeKeys = tf.estimator.ModeKeys
EstimatorSpec = tf.estimator.EstimatorSpec


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 125, 161, 2], name='cnn4')
  x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
  x = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')

  if params['verbose_summary']:
    tf.summary.image('input', x)

  conv1 = tf.layers.conv2d(x, filters=16, kernel_size=5, activation=tf.nn.relu, name='conv1')
  pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2, name='pool1')
  if params['verbose_summary']:
    log_conv_kernel('conv1')
    tf.summary.image('pool1', pool1[:, :, :, 0:1])

  conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=5, activation=tf.nn.relu, name='conv2')
  pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2, name='pool2')
  if params['verbose_summary']:
    log_conv_kernel('conv2')
    tf.summary.image('pool2', pool2[:, :, :, 0:1])

  conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=5, activation=tf.nn.relu, name='conv3')
  pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='pool3')
  if params['verbose_summary']:
    log_conv_kernel('conv3')
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

  conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv4')
  pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2, name='pool4')
  if params['verbose_summary']:
    log_conv_kernel('conv4')
    tf.summary.image('pool4', pool4[:, :, :, 0:1])

  conv5 = tf.layers.conv2d(pool4, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv5')
  pool5 = tf.layers.max_pooling2d(conv5, pool_size=[2, 2], strides=2, name='pool5')
  if params['verbose_summary']:
    log_conv_kernel('conv5')
    tf.summary.image('pool5', pool5[:, :, :, 0:1])

  conv = tf.layers.conv2d(pool5, filters=256, kernel_size=[1, 2], activation=tf.nn.relu, name='conv1x1')
  conv = tf.layers.conv2d(conv, filters=params['num_classes'], kernel_size=1, activation=None, name='conv_logits')
  logits = tf.reshape(conv, [-1, 12], name='flatten')


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
