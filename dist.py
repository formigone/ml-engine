import tensorflow as tf
import os
import sys
import json


def gen_input(filename):
  def decode(line):
    features = {
      'x': tf.FixedLenFeature((161 * 99,), tf.float32),
      'y': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(line, features)
    return parsed['x'], parsed['y']

  def input_fn():
    dataset = (tf.data.TFRecordDataset([filename])).map(decode)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(4)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return input_fn


def model_fn(features, labels, mode):
  x = tf.reshape(features, [-1, 99, 161, 1])

  conv = tf.layers.conv2d(x, filters=4, kernel_size=3, activation=tf.nn.relu)
  pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2)
  flat = tf.reshape(pool, [-1, 48 * 79 * 4])

  logits = tf.layers.dense(flat, units=12)

  predictions = {
    'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
    'probabilities': tf.nn.softmax(logits, name='prediction_probabilities'),
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=12, name='onehot_labels')
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  eval_metric_ops = {
    'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
  }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops
  )


def main(task_type, task_index, input, model_dir='/tmp/dist_home_train'):
  tf.logging.set_verbosity(tf.logging.DEBUG)
  input_fn = gen_input(input)

  cluster = {
    'cluster': {
      'master': ['10.0.0.23:2222'],
      'ps': ['10.0.0.23:2223'],
      'worker': ['10.0.0.42:2225'],
    },
    'task': {
      'type': task_type,
      'index': int(task_index),
    }
  }

  cluster = json.dumps(cluster)
  tf.logging.debug(' CONFIG: {}'.format(cluster))
  os.environ['TF_CONFIG'] = cluster
  config = tf.estimator.RunConfig(save_summary_steps=300, model_dir=model_dir)

  estimator = tf.estimator.Estimator(model_fn=model_fn, config=config)
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=10000)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=None, start_delay_secs=30, throttle_secs=30)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



main(sys.argv[1], sys.argv[2], sys.argv[3])