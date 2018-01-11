import tensorflow as tf

import runner
from util import inception_block, flatten


def model_fn(features, labels, mode, params):
  with tf.variable_scope('x_prep'):
    x = tf.reshape(features, [-1, 125, 161, 2], name='deception1')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    x = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')

  if params['verbose_summary']:
    tf.summary.image('input', x)

  conv = tf.layers.conv2d(x, filters=32, kernel_size=11, activation=tf.nn.relu, name='conv1')
  conv = tf.layers.conv2d(conv, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3')
  conv = tf.layers.conv2d(conv, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv4')
  conv = tf.layers.conv2d(conv, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv5')

  incep = inception_block(conv, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep6', norm=True, mode=mode)
  incep = inception_block(incep, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep7', norm=True, mode=mode)

  pool = tf.layers.max_pooling2d(incep, pool_size=[2, 2], strides=2, name='pool7')

  incep = inception_block(pool, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep8', norm=True, mode=mode)
  incep = inception_block(incep, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep9', norm=True, mode=mode)

  pool = tf.layers.max_pooling2d(incep, pool_size=[2, 2], strides=2, name='pool9')

  incep = inception_block(pool, t1x1=128, t3x3=128, t5x5=128, tmp=128, name='incep10', norm=True, mode=mode)

  pool = tf.layers.max_pooling2d(incep, pool_size=[2, 2], strides=2, name='pool11')

  flat = flatten(pool, name='flatten')
  dropout = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout')
  dense = tf.layers.dense(dropout, units=2048, activation=tf.nn.relu, name='dense')

  logits = tf.layers.dense(dense, units=params['num_classes'], name='logits')

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
