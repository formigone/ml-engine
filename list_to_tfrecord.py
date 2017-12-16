import tensorflow as tf
from tensorflow import python_io

tf.logging.set_verbosity(tf.logging.DEBUG)

def parse(input_file, output_file):
   tf.logging.debug('Input: {}'.format(input_file))
   with python_io.TFRecordWriter(output_file) as writer:
      with tf.gfile.GFile(input_file) as lines:
         for line in lines:
            line = line.strip().split(',')
            features = [float(val) for val in line[:-1]]
            label = int(line[-1])

            #tf.logging.debug('feat: {}'.format(len(features)))
            #tf.logging.debug('label: {}'.format(label))

            example = tf.train.Example()
            example.features.feature['x'].float_list.value.extend(features)
            example.features.feature['y'].int64_list.value.extend([label])
            writer.write(example.SerializeToString())

   tf.logging.debug('Output: {}'.format(output_file))


if __name__ == '__main__':
   FLAGS = tf.app.flags.FLAGS
   tf.app.flags.DEFINE_string('input_file', '', 'Input list to convert')
   tf.app.flags.DEFINE_string('output_file', '', 'Target output file')
   parse(FLAGS.input_file, FLAGS.output_file)

