import tensorflow as tf

import runner
from util import branch_incep, flatten, get_spectrogram


def model_fn(features, labels, mode, params):
  with tf.variable_scope('x_prep'):
    x = tf.reshape(features, [-1, 16000], name='input_incep107_flat_lg')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')

    spec_pow = get_spectrogram(x_norm, type='power', name='spec_pow')
    spec_mag = get_spectrogram(x_norm, type='magnitude', name='spec_mag')
    spec_mel = get_spectrogram(x_norm, type='mel', name='spec_mel')

  if params['verbose_summary']:
    tf.summary.audio('input', x_norm, 16000, max_outputs=12)
    tf.summary.image('mag', spec_mag)
    tf.summary.image('pow', spec_pow)
    tf.summary.image('mel', spec_mel)

  branch_pow = branch_incep(spec_pow, 'pow', params)
  branch_mag = branch_incep(spec_mag, 'mag', params)
  branch_mel = branch_incep(spec_mel, 'mel', params)

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
