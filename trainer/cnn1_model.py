import tensorflow as tf

ModeKeys = tf.estimator.ModeKeys
EstimatorSpec = tf.estimator.EstimatorSpec


def model_fn(features, labels, mode, params):
   x = tf.reshape(features, [-1, 200, 200, 3], name='input_cnn1')
   x_norm = tf.layers.batch_normalization(x, training=mode == ModeKeys.TRAIN, name='x_norm')

   tf.summary.image('input', x)

   conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, activation=tf.nn.relu, name='conv1')
   pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2, name='pool1')

   conv2 = tf.layers.conv2d(pool1, filters=32, kernel_size=3, activation=tf.nn.relu, name='conv2')
   pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2, name='pool2')

   conv3 = tf.layers.conv2d(pool2, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3')
   pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2,2], strides=2, name='pool3')

   dim = pool3.get_shape()[1:]
   dim = int(dim[0] * dim[1] * dim[2])
   flat = tf.reshape(pool3, [-1, dim], name='flat')

   dropout4 = tf.layers.dropout(flat, rate=params['dropout'], training=mode == ModeKeys.TRAIN, name='dropout4')
   dense4 = tf.layers.dense(dropout4, units=128, activation=tf.nn.relu, name='dense4')

   logits = tf.layers.dense(dense4, units=params['num_classes'], name='logits')

   predictions = {
      'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
      'probabilities': tf.nn.softmax(logits, name='prediction_probabilities'),
   }

   if mode == ModeKeys.PREDICT:
      return EstimatorSpec(mode=mode, predictions=predictions)

   onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['num_classes'], name='onehot_labels')
   loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
   tf.summary.scalar('loss', loss)

   optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
   with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
   eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes']) 
   }

   tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

   return EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops
   )

