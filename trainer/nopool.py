import tensorflow as tf

import runner
from util import inception_block, flatten
from graph_utils import log_conv_kernel


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_nopool')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    conv = x_norm
    conv = tf.layers.conv2d(conv, filters=96, kernel_size=5, activation=tf.nn.relu, name='conv1')
    conv = tf.layers.conv2d(conv, filters=96, kernel_size=3, activation=tf.nn.relu, name='conv2')
    conv = tf.layers.conv2d(conv, filters=96, kernel_size=3, activation=tf.nn.relu, name='conv3')
    conv = tf.layers.conv2d(conv, filters=96, kernel_size=3, activation=tf.nn.relu, name='conv4')
    fpool = tf.layers.conv2d(conv, filters=96, kernel_size=5, strides=(2, 2), activation=tf.nn.relu, name='faux_pool1')
    if params['verbose_summary']:
        log_conv_kernel('conv1')
        log_conv_kernel('conv2')
        log_conv_kernel('conv3')
        log_conv_kernel('conv4')
        log_conv_kernel('faux_pool1')

    conv = tf.layers.conv2d(fpool, filters=192, kernel_size=3, activation=tf.nn.relu, name='conv5')
    conv = tf.layers.conv2d(conv, filters=192, kernel_size=3, activation=tf.nn.relu, name='conv6')
    conv = tf.layers.conv2d(conv, filters=192, kernel_size=3, activation=tf.nn.relu, name='conv7')
    fpool = tf.layers.conv2d(conv, filters=192, kernel_size=5, strides=(2, 2), activation=tf.nn.relu, name='faux_pool2')
    if params['verbose_summary']:
        log_conv_kernel('conv5')
        log_conv_kernel('conv6')
        log_conv_kernel('conv7')
        log_conv_kernel('faux_pool2')

    conv = tf.layers.conv2d(fpool, filters=192, kernel_size=5, activation=tf.nn.relu, name='conv8')
    conv = tf.layers.conv2d(conv, filters=192, kernel_size=3, activation=tf.nn.relu, name='conv9')
    conv = tf.layers.conv2d(conv, filters=192, kernel_size=1, activation=tf.nn.relu, name='conv10')
    conv = tf.layers.conv2d(conv, filters=10, kernel_size=1, activation=tf.nn.relu, name='conv11')
    flat = flatten(conv)
    dense4 = tf.layers.dense(flat, units=1024, activation=tf.nn.relu, name='dense4')
    dropout4 = tf.layers.dropout(dense4, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout4')

    logits = tf.layers.dense(dropout4, units=params['num_classes'], name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
        'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=params['num_classes'], name='onehot_labels')
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    tf.summary.scalar('loss', loss)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=params['learning_rate'])
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }

    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == '__main__':
    runner.run(model_fn)
