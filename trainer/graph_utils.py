import tensorflow as tf


def log_conv_kernel(varname):
  conv_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}/kernel'.format(varname))
  tf.summary.histogram(varname, conv_kernel[0])
