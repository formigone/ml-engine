import tensorflow as tf


def log_conv_kernel(varname, prefix=''):
  conv_kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, '{}/kernel'.format(varname))
  if prefix is '':
    summary_name = '{}'.format(varname)
  else:
    summary_name = '{}/{}'.format(prefix, varname)
  tf.summary.histogram(summary_name, conv_kernel[0])
