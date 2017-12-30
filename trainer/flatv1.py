import tensorflow as tf

import graph_utils
from util import flatten
import runner


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 125, 128, 1], name='input_flatv1')
  x_flat = tf.reshape(features, [-1, 16000])
  x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
  if params['verbose_summary']:
    tf.summary.image('input', x)
    tf.summary.audio('input', x_flat, 16000)

  conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1')
  conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2')
  conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3')
  pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, name='pool3')
  if params['verbose_summary']:
    for i in range(1, 4):
      label = 'conv{}'.format(i)
      graph_utils.log_conv_kernel(label)
      tf.summary.image(label, tf.expand_dims(conv1[..., 0], -1))
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

  conv4 = tf.layers.conv2d(pool3, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv4')
  conv5 = tf.layers.conv2d(conv4, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv5')
  conv6 = tf.layers.conv2d(conv5, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv6')
  pool6 = tf.layers.max_pooling2d(conv6, pool_size=[2, 2], strides=2, name='pool6')
  if params['verbose_summary']:
    for i in range(4, 7):
      label = 'conv{}'.format(i)
      graph_utils.log_conv_kernel(label)
      tf.summary.image(label, tf.expand_dims(conv1[..., 0], -1))
    tf.summary.image('pool6', pool6[:, :, :, 0:1])

  conv7 = tf.layers.conv2d(pool6, filters=1024, kernel_size=3, activation=tf.nn.relu, name='conv7')
  conv8 = tf.layers.conv2d(conv7, filters=1024, kernel_size=5, activation=tf.nn.relu, name='conv8')
  conv9 = tf.layers.conv2d(conv8, filters=1024, kernel_size=7, activation=tf.nn.relu, name='conv9')
  pool9 = tf.layers.max_pooling2d(conv9, pool_size=[2, 2], strides=2, name='pool9')
  if params['verbose_summary']:
    for i in range(7, 10):
      label = 'conv{}'.format(i)
      graph_utils.log_conv_kernel(label)
      tf.summary.image(label, tf.expand_dims(conv1[..., 0], -1))
    tf.summary.image('pool9', pool9[:, :, :, 0:1])

  conv10 = tf.layers.conv2d(pool9, filters=512, kernel_size=1, activation=tf.nn.relu, name='conv10')
  conv11 = tf.layers.conv2d(conv10, filters=512, kernel_size=1, activation=tf.nn.relu, name='conv11')
  conv12 = tf.layers.conv2d(conv11, filters=512, kernel_size=1, activation=tf.nn.relu, name='conv12')

  flat = flatten(conv12)
  dense = tf.layers.dense(flat, units=1024, activation=tf.nn.relu, name='dense')

  logits = tf.layers.dense(dense, units=12, name='logits')

  predictions = {
    'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
    'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=12, name='onehot_labels')
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
