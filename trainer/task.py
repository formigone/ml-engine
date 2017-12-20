import tensorflow as tf

import cnn1_model, incep1_model, incep2_model, incep3_model, incep4_model, cnn2_model

FLAGS = tf.app.flags.FLAGS


def parse_args():
  flags = tf.app.flags
  flags.DEFINE_string('train_input', '', 'TFRecord used for training')
  flags.DEFINE_string('eval_input', '', 'TFRecord used for evaluation')
  flags.DEFINE_string('model_dir', '', 'Path to saved_model')
  flags.DEFINE_string('model', 'incep1', 'Model to be used')
  flags.DEFINE_float('learning_rate', 0.05, 'Learning rate')
  flags.DEFINE_float('dropout', 0.5, 'Dropout percentage')
  flags.DEFINE_integer('num_classes', 12, 'Number of classes to classify')
  flags.DEFINE_integer('epochs', 1, 'Total epocs to run')
  flags.DEFINE_integer('batch_size', 8, 'Input function batch size')
  flags.DEFINE_integer('buffer_size', 4, 'Input function buffer size')
  flags.DEFINE_string('verbosity', tf.logging.DEBUG, 'Verbosity level of Tensorflow app')

  flags.DEFINE_boolean('distributed', False, 'When set, be sure to also specify task_type and task_index')
  flags.DEFINE_string('task_type', '', 'Used in distributed training')
  flags.DEFINE_integer('task_index', -1, 'Used in distributed training')


def gen_input(filename, batch_size=16, repeat=1, buffer_size=1, record_shape=(161 * 99,)):
  tf.logging.debug('input_fn: {}'.format({
    'batch_size': batch_size,
    'repeat': repeat,
    'buffer_size': buffer_size,
    'shape': record_shape,
    'input': filename,
  }))

  def decode(line):
    features = {
      'x': tf.FixedLenFeature(record_shape, tf.float32),
      'y': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(line, features)
    return parsed['x'], parsed['y']

  def input_fn():
    dataset = (tf.data.TFRecordDataset([filename])).map(decode)
    if buffer_size > 1:
      dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.repeat(repeat)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return input_fn


def main(_):
  tf.logging.set_verbosity(FLAGS.verbosity)
  tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))
  tf.logging.debug('App config: {}'.format({
    'train_input': FLAGS.train_input,
    'eval_input': FLAGS.eval_input,
    'model': FLAGS.model,
    'model_dir': FLAGS.model_dir,
  }))
  train_input_fn = gen_input(FLAGS.train_input,
                       batch_size=FLAGS.batch_size,
                       repeat=FLAGS.epochs,
                       buffer_size=FLAGS.buffer_size)

  eval_input_fn = gen_input(FLAGS.eval_input)
  model_params = {
    'learning_rate': FLAGS.learning_rate,
    'dropout_rate': FLAGS.dropout,
    'num_classes': FLAGS.num_classes,
    'verbosity': FLAGS.verbosity,
    'verbose_summary': FLAGS.verbosity == tf.logging.DEBUG,
  }

  model_fn = None
  if FLAGS.model == 'cnn1':
    model_fn = cnn1_model.model_fn
  elif FLAGS.model == 'incep1':
    model_fn = incep1_model.model_fn
  elif FLAGS.model == 'incep2':
    model_fn = incep2_model.model_fn
  elif FLAGS.model == 'incep3':
    model_fn = incep3_model.model_fn
  elif FLAGS.model == 'incep4':
    model_fn = incep4_model.model_fn
  elif FLAGS.model == 'cnn2':
    model_fn = cnn2_model.model_fn

  if model_fn is None:
    raise ValueError('Invalid module_fn')

  tf.logging.debug('params: {}'.format(model_params))

  estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn, params=model_params)
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, start_delay_secs=30, throttle_secs=30)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
  parse_args()
  tf.app.run(main)
