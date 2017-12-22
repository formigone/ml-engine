import tensorflow as tf


def parse_args():
  flags = tf.app.flags
  flags.DEFINE_string('train_input', '', 'TFRecord used for training')
  flags.DEFINE_string('eval_input', '', 'TFRecord used for evaluation')
  flags.DEFINE_string('model_dir', '', 'Path to saved_model')
  flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
  flags.DEFINE_float('dropout', 0.5, 'Dropout percentage')
  flags.DEFINE_integer('num_classes', 12, 'Number of classes to classify')
  flags.DEFINE_integer('batch_size', 32, 'Input function batch size')
  flags.DEFINE_integer('buffer_size', 256, 'Input function buffer size')
  flags.DEFINE_integer('repeat_training', 1, 'How many times to repeat entire training sets')
  flags.DEFINE_string('verbosity', tf.logging.DEBUG, 'Logging verbosity level')

  return flags.FLAGS


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


def run(model_fn):
  args = parse_args()
  tf.logging.set_verbosity(args.verbosity)
  tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))
  tf.logging.debug('App config: {}'.format({
    'train_input': args.train_input,
    'eval_input': args.eval_input,
    'model_dir': args.model_dir,
  }))

  train_input_fn = gen_input(args.train_input, batch_size=args.batch_size, buffer_size=args.buffer_size, repeat=args.repeat_training)
  eval_input_fn = gen_input(args.eval_input)

  model_params = {
    'learning_rate': args.learning_rate,
    'dropout_rate': args.dropout,
    'num_classes': args.num_classes,
    'verbosity': args.verbosity,
    'verbose_summary': args.verbosity == tf.logging.DEBUG,
  }

  tf.logging.debug('params: {}'.format(model_params))

  estimator = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=model_fn, params=model_params)
  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, start_delay_secs=30)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
