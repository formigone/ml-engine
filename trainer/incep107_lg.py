import tensorflow as tf

import runner
from util import inception_block, flatten
from graph_utils import log_conv_kernel


def get_spectrogram(samples, type='power', sr=16000, frame_len=256, step=64, fft_len=512, low=80.0, high=7600.0,
                    bins=257, name='spec'):
  """
  Returns a MxN spectrogram of a given type
  :param samples: Tensor of shape [batch_size, samples]
  :param type: Either 'power', 'magnitue', or 'mel'
  :param sr:
  :param frame_len:
  :param step:
  :param fft_len:
  :param low:
  :param high:
  :param bins:
  :return:
  """
  stfts = tf.contrib.signal.stft(samples, frame_length=frame_len, frame_step=step, fft_length=fft_len)

  if type == 'power':
    spec = tf.real(stfts * tf.conj(stfts))
  elif type == 'magnitude':
    spec = tf.abs(stfts)
  elif type == 'mel':
    spec = tf.abs(stfts)
    num_spectrogram_bins = spec.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = low, high, bins
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spec, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spec.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    spec = mel_spectrograms
  else:
    raise Exception('Invalid spectrogram type "{}"'.format(type))

  log_offset = 1e-6
  log_spec = tf.log(spec + log_offset)

  _, spec_h, spec_w = log_spec.get_shape()

  return tf.reshape(log_spec, [-1, spec_h, spec_w, 1], name=name)


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


def model_fn(features, labels, mode, params):
  with tf.variable_scope('x_prep'):
    x = tf.reshape(features, [-1, 125, 161, 2], name='input_incep107_lg')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    x_spec = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')
    x_freq = tf.reshape(x_norm[:, :, :, 1], [-1, 125, 161], name='reshape_freq')
    x_freq = tf.slice(x_freq, [0, 0, 0], [-1, 125, 128], name='slice_freq')
    x_freq = tf.reshape(x_freq, [-1, 125 * 128], name='flatten_freq')

    spec_pow = get_spectrogram(x_freq, type='power', name='spec_pow')
    spec_mag = get_spectrogram(x_freq, type='magnitude', name='spec_mag')
    spec_mel = get_spectrogram(x_freq, type='mel', name='spec_mel')

  if params['verbose_summary']:
    tf.summary.audio('input', x_freq, 16000, max_outputs=12)
    tf.summary.image('input', x_spec)
    tf.summary.image('mag', spec_mag)
    tf.summary.image('pow', spec_pow)
    tf.summary.image('mel', spec_mel)

  branch_pow = branch_incep(spec_pow, 'pow', mode, params)
  branch_mag = branch_incep(spec_mag, 'mag', mode, params)
  branch_mel = branch_incep(spec_mel, 'mel', mode, params)

  with tf.variable_scope('concat'):
    flat_pow = flatten(branch_pow, 'flat_pow')
    flat_mag = flatten(branch_mag, 'flat_mag')
    flat_mel = flatten(branch_mel, 'flat_mel')
    branch_merge = tf.concat([flat_pow, flat_mag, flat_mel], axis=1, name='merge')
  norm = tf.layers.batch_normalization(branch_merge, training=mode == tf.estimator.ModeKeys.TRAIN, name='norm_merge')
  dense = tf.layers.dense(norm, units=2048, activation=tf.nn.relu, name='dense')
  dropout = tf.layers.dropout(dense, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN,
                              name='dropout')

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
