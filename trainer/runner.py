import tensorflow as tf
import numpy as np


classes = [
    'up',       # 0
    'down',     # 1
    'left',     # 2
    'right',    # 3
    'go',       # 4
    'stop',     # 5
    'yes',      # 6
    'no',       # 7
    'on',       # 8
    'off',      # 9
    'silence',  # 10
    'unknown',  # 11
]

label_key_map = {k: v for v, k in enumerate(classes)}
key_label_map = {v: k for v, k in enumerate(classes)}


def int2label(value, default='unknown'):
    try:
        return key_label_map[value]
    except KeyError:
        return default


def label2int(value, default=11, v2=False):
    try:
        if v2 and value == 'noise':
            raise ValueError('BOOM')
        return label_key_map[value]
    except KeyError:
        return default

def parse_args():
  flags = tf.app.flags
  flags.DEFINE_string('train_input', '', 'TFRecord used for training')
  flags.DEFINE_string('eval_input', '', 'TFRecord used for evaluation')
  flags.DEFINE_string('predict_input', '', 'TFRecord used for prediction')
  flags.DEFINE_string('model_dir', '', 'Path to saved_model')
  flags.DEFINE_string('mode', 'train', 'Either train or predict')
  flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
  flags.DEFINE_float('dropout', 0.5, 'Dropout percentage')
  flags.DEFINE_integer('num_classes', 12, 'Number of classes to classify')
  flags.DEFINE_integer('batch_size', 32, 'Input function batch size')
  flags.DEFINE_integer('buffer_size', 128, 'Input function buffer size')
  flags.DEFINE_integer('repeat_training', 1, 'How many times to repeat entire training sets')
  flags.DEFINE_string('verbosity', tf.logging.DEBUG, 'Logging verbosity level')
  flags.DEFINE_string('input_shape', 'stack', 'Either "stack" (125,161,2), "flat" (16000,), or "spec" (99,161)')

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
    'input_shape': args.input_shape,
  }))

  if args.input_shape == 'flat':
    input_shape = (16000,)
  elif args.input_shape == 'spec':
    input_shape = (161 * 99,)
  elif args.input_shape == 'stack':
    input_shape = (125 * 161 * 2,)
  else:
    raise ValueError('Invalid input_shape')

  model_params = {
    'learning_rate': args.learning_rate,
    'dropout_rate': args.dropout,
    'num_classes': args.num_classes,
    'verbosity': args.verbosity,
    'verbose_summary': args.verbosity == tf.logging.DEBUG,
  }

  tf.logging.debug('params: {}'.format(model_params))

  estimator = tf.estimator.Estimator(model_dir=args.model_dir, model_fn=model_fn, params=model_params)

  if args.mode == 'train':
    train_input_fn = gen_input(args.train_input,
                               batch_size=args.batch_size,
                               buffer_size=args.buffer_size,
                               repeat=args.repeat_training,
                               record_shape=input_shape)
    eval_input_fn = gen_input(args.eval_input, record_shape=input_shape)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=None)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None, start_delay_secs=30)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  elif args.mode == 'predict':
    input_fn = gen_input(args.predict_input, record_shape=input_shape)
    predictions = estimator.predict(input_fn=input_fn)

    for pred in predictions:
      label = int2label(np.argmax(pred['predictions']))
      print('{} ==> {}'.format(label, pred['predictions']))
  else:
    raise ValueError('Invalid mode')
