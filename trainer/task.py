import tensorflow as tf

from models import cnn1

tf.logging.set_verbosity('DEBUG')
tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))

FLAGS = tf.app.flags.FLAGS

def parse_args():
   flags = tf.app.flags
   flags.DEFINE_string('input', '', 'Input csv file')
   flags.DEFINE_string('mode', 'train', 'Either train, eval, predict')
   flags.DEFINE_string('model_dir', '', 'Path to saved_model')
   flags.DEFINE_string('model', '', 'Model to be used')
   flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
   flags.DEFINE_float('dropout', 0.5, 'Dropout percentage')
   flags.DEFINE_integer('num_classes', 3, 'Number of classes to classify')
   flags.DEFINE_integer('epochs', 1, 'Total epocs to run')
   flags.DEFINE_integer('batch_size', 16, 'Input function batch size')
   flags.DEFINE_integer('buffer_size', 16, 'Input function buffer size')
   flags.DEFINE_boolean('shuffle', False, 'Should input_fn shuffle input')
   flags.DEFINE_string('verbosity', tf.logging.DEBUG, 'Verbosity level of Tensorflow app')


def gen_input(filename, batch_size=16, shuffle=False, repeat=1, buffer_size=16, record_shape=(4,)):
   tf.logging.debug('input_fn: {}'.format({'batch_size': batch_size, 'shuffle': shuffle, 'repeat': repeat, 'buffer_size': buffer_size}))
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
   tf.logging.debug('... {}'.format(FLAGS.input))
   input_fn = gen_input(FLAGS.input, 
      batch_size=FLAGS.batch_size, 
      repeat=FLAGS.epochs, 
      shuffle=FLAGS.shuffle, 
      buffer_size=FLAGS.buffer_size, 
      record_shape=(200 * 200 * 3,)
   )
   model_params = {
      'learning_rate': FLAGS.learning_rate,
      'dropout': FLAGS.dropout,
      'num_classes': FLAGS.num_classes,
      'verbosity': FLAGS.verbosity,
   }

   model_fn = None
   if FLAGS.model == '':
      model_fn = cnn1.model_fn

   tf.logging.debug('params: {}'.format(model_params))
   estimator = tf.estimator.Estimator(model_dir=FLAGS.model_dir, model_fn=model_fn, params=model_params)

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
         record_shape=(200 * 200 * 3,)
      )

      predictions = estimator.predict(input_fn=input_fn)
      for pred in predictions:
         tf.logging.info('Pred: {}'.format(pred))


if __name__ == '__main__':
   parse_args()
   tf.app.run(main)

