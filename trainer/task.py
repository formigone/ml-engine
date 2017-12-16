import tensorflow as tf

tf.logging.set_verbosity('DEBUG')
tf.logging.debug('Tensorflow version: {}'.format(tf.__version__))

FLAGS = tf.app.flags.FLAGS

def parse_args():
   flags = tf.app.flags
   flags.DEFINE_string('input', '', 'Input csv file')


def gen_input(filename, batch_size=16, shuffle=False, repeat=1, buffer_size=16, record_shape=(4,)):
   def decode(line):
      features = {
         'x': tf.FixedLenFeature(record_shape, tf.float32),
         'y': tf.FixedLenFeature((), tf.int64)
      }
      parsed = tf.parse_single_example(line, features)
      return parsed['x'], parsed['y']

   dataset = (tf.data.TFRecordDataset([filename])).map(decode)
   if shuffle:
      dataset = dataset.shuffle(buffer_size=buffer_size)
   dataset = dataset.repeat(repeat)
   dataset = dataset.batch(batch_size)
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

def main(_):
   tf.logging.debug('... {}'.format(FLAGS.input))
   input = gen_input(FLAGS.input, batch_size=6, repeat=3, shuffle=True, buffer_size=12, record_shape=(200 * 200 * 3,))
   with tf.Session() as sess:
      while True:
         try:
            feat, labels = sess.run(input)
            tf.logging.debug(labels)
         except:
            break



if __name__ == '__main__':
   parse_args()
   tf.app.run(main)

