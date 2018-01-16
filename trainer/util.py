import tensorflow as tf

from graph_utils import log_conv_kernel


def flatten(input, name='flatten'):
  dim = input.get_shape()[1:]
  dim = int(dim[0] * dim[1] * dim[2])
  return tf.reshape(input, [-1, dim], name=name)


def touch_incep(input, name):
  dim = input.get_shape()[1:]
  return tf.reshape(input, [-1, dim[0], dim[1], dim[2]], name=name)


def double_inception(prev, block_depth, name):
  tower = inception_block(prev, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth, name=name)
  tower_res = touch_incep(tower, '{}_res'.format(name))
  return inception_block(tower_res, t1x1=block_depth, t3x3=block_depth, t5x5=block_depth, tmp=block_depth,
                         name='{}_2'.format(name))


def conv_group(prev, filters, name='conv_group', verbose=False):
  with tf.variable_scope(name):
    conv = tf.layers.conv2d(prev, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu,
                            name='conv_same')
    convb = tf.layers.conv2d(conv, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convb')
    convc = tf.layers.conv2d(convb, filters=filters, kernel_size=3, activation=tf.nn.relu, name='convc')
    pool = tf.layers.max_pooling2d(convc, pool_size=[2, 2], strides=2, name='pool')
    if verbose:
      log_conv_kernel('{}/conv_same'.format(name))
      log_conv_kernel('{}/convb'.format(name))
      log_conv_kernel('{}/convc'.format(name))
      tf.summary.image('{}/pool'.format(name), pool[:, :, :, 0:1])
    return pool


def inception_block(prev, t1x1=2, t3x3=2, t5x5=2, tmp=2, name='incep', norm=False, mode=None):
  training = mode == tf.estimator.ModeKeys.TRAIN
  with tf.variable_scope(name):
    with tf.variable_scope('1x1_conv'):
      tower_1x1 = tf.layers.conv2d(prev,
                                   filters=t1x1,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      if norm and mode is not None:
        tower_1x1 = tf.layers.batch_normalization(tower_1x1, training=training, name='batch_norm')

    with tf.variable_scope('3x3_conv'):
      tower_3x3 = tf.layers.conv2d(prev,
                                   filters=t3x3,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      tower_3x3 = tf.layers.conv2d(tower_3x3,
                                   filters=t3x3,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv')
      if norm and mode is not None:
        tower_3x3 = tf.layers.batch_normalization(tower_3x3, training=training, name='batch_norm')

    with tf.variable_scope('5x5_conv'):
      tower_5x5 = tf.layers.conv2d(prev,
                                   filters=t5x5,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      tower_5x5 = tf.layers.conv2d(tower_5x5,
                                   filters=t5x5,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv_1')
      tower_5x5 = tf.layers.conv2d(tower_5x5,
                                   filters=t5x5,
                                   kernel_size=3,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='3x3_conv_2')
      if norm and mode is not None:
        tower_5x5 = tf.layers.batch_normalization(tower_5x5, training=training, name='batch_norm')

    with tf.variable_scope('maxpool'):
      tower_mp = tf.layers.max_pooling2d(prev,
                                         pool_size=3,
                                         strides=1,
                                         padding='same',
                                         name='3x3_maxpool')
      tower_mp = tf.layers.conv2d(tower_mp,
                                  filters=tmp,
                                  kernel_size=1,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='1x1_conv')

      if norm and mode is not None:
        tower_mp = tf.layers.batch_normalization(tower_mp, training=training, name='batch_norm')

    return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)


def inception_block_v2(prev, t1x1=2, t3x3=2, t5x5=2, tmp=2, name='incep', norm=False, mode=None):
  with tf.variable_scope(name):
    with tf.variable_scope('1x1_conv'):
      tower_1x1 = tf.layers.conv2d(prev,
                                   filters=t1x1,
                                   kernel_size=1,
                                   padding='same',
                                   activation=tf.nn.relu,
                                   name='1x1_conv')
      if norm and mode is not None:
        tower_1x1 = tf.layers.batch_normalization(tower_1x1, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                  name='batch_norm')

    with tf.variable_scope('3x3_conv'):
      tower_3x3_1x1 = tf.layers.conv2d(prev,
                                       filters=t3x3,
                                       kernel_size=1,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='1x1_conv')
      tower_3x3_1x3 = tf.layers.conv2d(tower_3x3_1x1,
                                       filters=t3x3,
                                       kernel_size=(1, 3),
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='1x3_conv')
      tower_3x3_3x1 = tf.layers.conv2d(tower_3x3_1x1,
                                       filters=t3x3,
                                       kernel_size=(3, 1),
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='3x1_conv')
      if norm and mode is not None:
        tower_3x3_1x3 = tf.layers.batch_normalization(tower_3x3_1x3, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                      name='batch_norm_1x3')
        tower_3x3_3x1 = tf.layers.batch_normalization(tower_3x3_3x1, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                      name='batch_norm_3x1')

    with tf.variable_scope('5x5_conv'):
      tower_5x5_1x1 = tf.layers.conv2d(prev,
                                       filters=t5x5,
                                       kernel_size=1,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='1x1_conv')
      tower_5x5_3x3 = tf.layers.conv2d(tower_5x5_1x1,
                                       filters=t5x5,
                                       kernel_size=3,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='3x3_conv')
      tower_5x5_1x3 = tf.layers.conv2d(tower_5x5_3x3,
                                       filters=t5x5,
                                       kernel_size=3,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='1x3_conv')
      tower_5x5_3x1 = tf.layers.conv2d(tower_5x5_3x3,
                                       filters=t5x5,
                                       kernel_size=3,
                                       padding='same',
                                       activation=tf.nn.relu,
                                       name='3x1_conv')
      if norm and mode is not None:
        tower_5x5_1x3 = tf.layers.batch_normalization(tower_5x5_1x3, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                      name='batch_norm_1x3')
        tower_5x5_3x1 = tf.layers.batch_normalization(tower_5x5_3x1, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                      name='batch_norm_3x1')

    with tf.variable_scope('maxpool'):
      tower_mp = tf.layers.max_pooling2d(prev,
                                         pool_size=3,
                                         strides=1,
                                         padding='same',
                                         name='3x3_maxpool')
      tower_mp = tf.layers.conv2d(tower_mp,
                                  filters=tmp,
                                  kernel_size=1,
                                  padding='same',
                                  activation=tf.nn.relu,
                                  name='1x1_conv')
      if norm and mode is not None:
        tower_mp = tf.layers.batch_normalization(tower_mp, training=mode == tf.estimator.ModeKeys.TRAIN,
                                                 name='batch_norm')

    return tf.concat([tower_1x1, tower_3x3_1x3, tower_3x3_3x1, tower_5x5_1x3, tower_5x5_3x1, tower_mp], axis=3)


def branch_incep(x, name, params):
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


def get_spectrogram(samples, type='power', sr=16000, frame_len=256, step=64, fft_len=512, low=80.0, high=7600.0,
                    bins=257, reshape_flat=False, name='spec'):
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
  elif type == 'mel/pow':
    spec = tf.real(stfts * tf.conj(stfts))
    num_spectrogram_bins = spec.shape[-1].value
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = low, high, bins
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
      num_mel_bins, num_spectrogram_bins, sr, lower_edge_hertz, upper_edge_hertz)
    mel_spectrograms = tf.tensordot(spec, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(spec.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    spec = mel_spectrograms
  elif type == 'mel/mag' or type == 'mel':
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

  if reshape_flat:
    return tf.reshape(log_spec, [-1, spec_h * spec_w], name=name)

  return tf.reshape(log_spec, [-1, spec_h, spec_w, 1], name=name)


def conf_mat(labels, predictions, num_classes):
  with tf.variable_scope('confusion_matrix'):
    confusion_matrix = tf.confusion_matrix(labels, predictions, num_classes=num_classes)
    confusion_matrix = tf.reshape(confusion_matrix, [1, num_classes, num_classes, 1])
    return tf.cast(confusion_matrix, dtype=tf.float32)
