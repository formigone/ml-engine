import tensorflow as tf

import runner
from util import inception_block, log_conv_kernel, flatten, get_spectrogram


def model_fn(features, labels, mode, params):
  with tf.variable_scope('x_prep'):
    x = tf.reshape(features, [-1, 16000], name='input_incep109_lg')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')

    spec_pow = get_spectrogram(x_norm, type='power', reshape_flat=True, name='spec_pow')
    spec_mag = get_spectrogram(x_norm, type='magnitude', reshape_flat=True, name='spec_mag')
    spec_mel = get_spectrogram(x_norm, type='mel', reshape_flat=True, name='spec_mel')

    _, h, w, _ = [-1, 247, 257, 1]
    spec = tf.reshape(tf.stack([spec_pow, spec_mag, spec_mel], name='stack'), [-1, h, w, 3], name='stack_reshape')

  if params['verbose_summary']:
    tf.summary.audio('input', x_norm, 16000, max_outputs=12)
    tf.summary.image('mag', spec_mag)
    tf.summary.image('pow', spec_pow)
    tf.summary.image('mel', spec_mel)
    tf.summary.image('spec', spec)

  conv1 = tf.layers.conv2d(spec, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
  conv1b = tf.layers.conv2d(conv1, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1b')
  conv1c = tf.layers.conv2d(conv1b, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1c')
  pool1 = tf.layers.max_pooling2d(conv1c, pool_size=[2, 2], strides=2, name='pool1')
  if params['verbose_summary']:
    log_conv_kernel('conv1')
    log_conv_kernel('conv1b')
    log_conv_kernel('conv1c')
    tf.summary.image('pool1', pool1[:, :, :, 0:1])

  incep2 = inception_block(pool1, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep2')

  conv3 = tf.layers.conv2d(incep2, filters=32, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
  conv3b = tf.layers.conv2d(conv3, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv3b')
  conv3c = tf.layers.conv2d(conv3b, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv3c')
  pool3 = tf.layers.max_pooling2d(conv3c, pool_size=[2, 2], strides=2, name='pool3')
  if params['verbose_summary']:
    log_conv_kernel('conv3')
    log_conv_kernel('conv3b')
    log_conv_kernel('conv3c')
    tf.summary.image('pool3', pool3[:, :, :, 0:1])

  conv5 = tf.layers.conv2d(pool3, filters=64, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5')
  conv5b = tf.layers.conv2d(conv5, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv5b')
  conv5c = tf.layers.conv2d(conv5b, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv5c')
  pool5 = tf.layers.max_pooling2d(conv5c, pool_size=[2, 2], strides=2, name='pool5')
  if params['verbose_summary']:
    log_conv_kernel('conv5')
    log_conv_kernel('conv5b')
    log_conv_kernel('conv5c')
    tf.summary.image('pool5', pool5[:, :, :, 0:1])

  incep6 = inception_block(pool5, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep6')

  conv7 = tf.layers.conv2d(incep6, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7')
  conv7b = tf.layers.conv2d(conv7, filters=128, kernel_size=3, activation=tf.nn.relu, name='conv7b')
  pool7 = tf.layers.max_pooling2d(conv7b, pool_size=[2, 2], strides=2, name='pool7')
  if params['verbose_summary']:
    log_conv_kernel('conv7')
    log_conv_kernel('conv7b')
    tf.summary.image('pool7', pool7[:, :, :, 0:1])

  conv8 = tf.layers.conv2d(pool7, filters=256, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv8')
  conv8b = tf.layers.conv2d(conv8, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv8b')
  conv8c = tf.layers.conv2d(conv8b, filters=256, kernel_size=3, activation=tf.nn.relu, name='conv8c')
  if params['verbose_summary']:
    log_conv_kernel('conv8')
    log_conv_kernel('conv8b')
    log_conv_kernel('conv8c')

  incep9 = inception_block(conv8c, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep9')

  conv10 = tf.layers.conv2d(incep9, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv10')
  pool10 = tf.layers.max_pooling2d(conv10, pool_size=[2, 2], strides=2, name='pool10')
  if params['verbose_summary']:
    log_conv_kernel('conv10')
    tf.summary.image('pool8', pool10[:, :, :, 0:1])

  flat = flatten(pool10)
  dense4 = tf.layers.dense(flat, units=2048, activation=tf.nn.relu, name='dense4')
  dropout4 = tf.layers.dropout(dense4, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN,
                               name='dropout4')

  logits = tf.layers.dense(dropout4, units=params['num_classes'], name='logits')

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
