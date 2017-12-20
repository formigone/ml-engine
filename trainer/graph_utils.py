import tensorflow as tf


def log_conv_kernel(varname, prefix=''):
  conv_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}/kernel'.format(varname))
  tf.summary.histogram('{}/{}'.format(prefix, varname), conv_kernel[0])
