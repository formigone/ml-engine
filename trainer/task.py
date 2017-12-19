import tensorflow as tf

import cnn1_model, incep1_model

FLAGS = tf.app.flags.FLAGS


def parse_args():
  flags = tf.app.flags
  flags.DEFINE_string('input', '', 'Input csv file')
  flags.DEFINE_string('mode', 'train', 'Either train, eval, predict')
  flags.DEFINE_string('model_dir', '', 'Path to saved_model')
  flags.DEFINE_string('model', 'incep1', 'Model to be used')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_float('dropout', 0.5, 'Dropout percentage')
  flags.DEFINE_integer('num_classes', 12, 'Number of classes to classify')
  flags.DEFINE_integer('epochs', 1, 'Total epocs to run')
  flags.DEFINE_integer('batch_size', 8, 'Input function batch size')
  flags.DEFINE_integer('buffer_size', 4, 'Input function buffer size')
  flags.DEFINE_boolean('shuffle', False, 'Should input_fn shuffle input')
  flags.DEFINE_string('verbosity', tf.logging.DEBUG, 'Verbosity level of Tensorflow app')


def gen_input(filename, batch_size=16, shuffle=False, repeat=1, buffer_size=16, record_shape=(4,)):
  tf.logging.debug('input_fn: {}'.format(
    {'batch_size': batch_size, 'shuffle': shuffle, 'repeat': repeat, 'buffer_size': buffer_size,
     'shape': record_shape}))

  def decode(line):
    features = {
      'x': tf.FixedLenFeature(record_shape, tf.float32),
      'y': tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(line, features)
    return parsed['x'], parsed['y']

  def input_fn():
    dataset = (tf.data.TFRecordDataset([filename])).map(decode)
    if shuffle:
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
  tf.logging.debug('App config: {}'.format(
    {'input': FLAGS.input, 'mode': FLAGS.mode, 'model': FLAGS.model, 'model_dir': FLAGS.model_dir}))
  input_fn = gen_input(FLAGS.input,
                       batch_size=FLAGS.batch_size,
                       repeat=FLAGS.epochs,
                       shuffle=FLAGS.shuffle,
                       buffer_size=FLAGS.buffer_size,
                       record_shape=(161 * 99,)
                       )
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

  if model_fn is None:
    raise ValueError('Invalid module_fn')

  tf.logging.debug('params: {}'.format(model_params))
  run_config = tf.estimator.RunConfig()
  run_config = run_config.replace(model_dir=FLAGS.model_dir)
  estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn, params=model_params,
                                     config=run_config)

  if FLAGS.mode == 'train':
    estimator.train(input_fn=input_fn)
  elif FLAGS.mode == 'eval':
    estimator.evaluate(input_fn=input_fn)
  elif FLAGS.mode == 'predict':
    input_fn = gen_input(FLAGS.input,
                         batch_size=FLAGS.batch_size,
                         repeat=1,
                         shuffle=False,
                         buffer_size=FLAGS.buffer_size,
                         record_shape=(161 * 99,)
                         )

    predictions = estimator.predict(input_fn=input_fn)
    for pred in predictions:
      tf.logging.info('Pred: {}'.format(pred))


if __name__ == '__main__':
  parse_args()
  tf.app.run(main)
