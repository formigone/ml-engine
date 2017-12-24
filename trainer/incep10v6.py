import tensorflow as tf

import runner
from util import inception_block, flatten
from graph_utils import log_conv_kernel


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep10v6')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    conv1 = tf.layers.conv2d(x_norm, filters=16, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
    conv1b = tf.layers.conv2d(conv1, filters=24, kernel_size=3, activation=tf.nn.relu, name='conv1b')
    pool1 = tf.layers.max_pooling2d(conv1b, pool_size=[2, 2], strides=2, name='pool1')
    if params['verbose_summary']:
        log_conv_kernel('conv1')
        log_conv_kernel('conv1b')
        tf.summary.image('pool1', pool1[:, :, :, 0:1])

    incep2 = inception_block(pool1, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep2', norm=True, mode=mode)

    conv3 = tf.layers.conv2d(incep2, filters=48, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv3')
    conv3b = tf.layers.conv2d(conv3, filters=64, kernel_size=3, activation=tf.nn.relu, name='conv3b')
    pool3 = tf.layers.max_pooling2d(conv3b, pool_size=[2, 2], strides=2, name='pool3')
    if params['verbose_summary']:
        log_conv_kernel('conv3')
        log_conv_kernel('conv3b')
        tf.summary.image('pool3', pool3[:, :, :, 0:1])

    incep4 = inception_block(pool3, t1x1=24, t3x3=24, t5x5=24, tmp=24, name='incep4', norm=True, mode=mode)

    conv5 = tf.layers.conv2d(incep4, filters=128, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv5')
    conv5b = tf.layers.conv2d(conv5, filters=192, kernel_size=3, activation=tf.nn.relu, name='conv5b')
    pool5 = tf.layers.max_pooling2d(conv5b, pool_size=[2, 2], strides=2, name='pool5')
    if params['verbose_summary']:
        log_conv_kernel('conv5')
        log_conv_kernel('conv5b')
        tf.summary.image('pool5', pool5[:, :, :, 0:1])

    incep6 = inception_block(pool5, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep6', norm=True, mode=mode)

    conv7 = tf.layers.conv2d(incep6, filters=384, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv7')
    conv7b = tf.layers.conv2d(conv7, filters=512, kernel_size=3, activation=tf.nn.relu, name='conv7b')
    pool7 = tf.layers.max_pooling2d(conv7b, pool_size=[2, 2], strides=2, name='pool7')
    if params['verbose_summary']:
        log_conv_kernel('conv7')
        log_conv_kernel('conv7b')
        tf.summary.image('pool7', pool7[:, :, :, 0:1])

    incep8 = inception_block(pool7, t1x1=192, t3x3=192, t5x5=192, tmp=192, name='incep8', norm=True, mode=mode)

    flat = flatten(incep8)
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
