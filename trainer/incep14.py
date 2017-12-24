import tensorflow as tf

import runner
from util import inception_block_v2, flatten
from graph_utils import log_conv_kernel


def model_fn(features, labels, mode, params):
  x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep14')
  x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
  if params['verbose_summary']:
    tf.summary.image('input', x)

  conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1')
  conv2 = tf.layers.conv2d(conv1, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2')
  conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu, name='conv3')
  if params['verbose_summary']:
    log_conv_kernel('conv1')
    log_conv_kernel('conv2')
    log_conv_kernel('conv3')
    tf.summary.image('conv3', conv3[:, :, :, 0:1])

  conv4 = tf.layers.conv2d(conv3, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv4')
  conv5 = tf.layers.conv2d(conv4, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv5')
  conv6 = tf.layers.conv2d(conv5, filters=128, kernel_size=3, strides=(2, 2), activation=tf.nn.relu, name='conv6')
  if params['verbose_summary']:
    log_conv_kernel('conv4')
    log_conv_kernel('conv5')
    log_conv_kernel('conv6')
    tf.summary.image('conv6', conv6[:, :, :, 0:1])

  incep4 = inception_block_v2(conv6, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep4', norm=True, mode=mode)
  incep5 = inception_block_v2(incep4, t1x1=48, t3x3=48, t5x5=48, tmp=48, name='incep5', norm=True, mode=mode)
  incep6 = inception_block_v2(incep5, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep6', norm=True, mode=mode)
  # incep7 = inception_block_v2(incep6, t1x1=96, t3x3=96, t5x5=96, tmp=96, name='incep7', norm=True, mode=mode)
  incep8 = inception_block_v2(incep6, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep8', norm=True, mode=mode)
  # incep9 = inception_block_v2(incep8, t1x1=192, t3x3=192, t5x5=192, tmp=192, name='incep9', norm=True, mode=mode)
  incep10 = inception_block_v2(incep8, t1x1=256, t3x3=256, t5x5=256, tmp=256, name='incep10', norm=True, mode=mode)
  # incep11 = inception_block_v2(incep10, t1x1=512, t3x3=512, t5x5=512, tmp=512, name='incep11', norm=True, mode=mode)
  # incep12 = inception_block_v2(incep11, t1x1=768, t3x3=768, t5x5=768, tmp=768, name='incep12', norm=True, mode=mode)

  flat = flatten(incep10)
  dense13 = tf.layers.dense(flat, units=2048, activation=tf.nn.relu, name='dense13')
  dropout13 = tf.layers.dropout(dense13, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout13')

  logits = tf.layers.dense(dropout13, units=params['num_classes'], name='logits')

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
