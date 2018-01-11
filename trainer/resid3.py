import tensorflow as tf

import runner
from util import flatten


def resid_block(prev, filters, mode, name):
    with tf.variable_scope(name):
        conv1 = tf.layers.conv2d(prev, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv1')
        conv = tf.layers.conv2d(conv1, filters=filters, kernel_size=3, padding='same', activation=tf.nn.relu, name='conv2')
        conv = tf.layers.conv2d(conv, filters=filters, kernel_size=3, padding='same', activation=None, name='conv3')
        conv = tf.nn.relu(conv + conv1, name='relu')
        conv = tf.layers.batch_normalization(conv, training=mode == tf.estimator.ModeKeys.TRAIN, name='batch_norm')
    return conv


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 125, 161, 2], name='redid3')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    x = tf.reshape(x_norm[:, :, :, 0], [-1, 125, 161, 1], name='reshape_spec')

    conv = x
    l = 1
    for i in [64, 128, 256, 512]:
        conv = resid_block(conv, i, mode, 'conv_{}'.format(l))
        l += 1

    flat = flatten(conv)
    logits = tf.layers.dense(flat, units=params['num_classes'], name='logits')

    predictions = {
        'classes': tf.argmax(logits, axis=1, name='prediction_classes'),
        'probabilities': tf.nn.softmax(logits, name='prediction_softmax')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={'predictions': predictions['probabilities']})

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

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


if __name__ == '__main__':
    runner.run(model_fn)
