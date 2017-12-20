import tensorflow as tf

from util import inception_block, flatten


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep2')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    incep1 = inception_block(x_norm, name='incep1')
    incep2 = inception_block(incep1, t1x1=4, t3x3=4, t5x5=4, tmp=4, name='incep2')
    incep3 = inception_block(incep2, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep3')

    flat = flatten(incep3)
    dropout4 = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout4')
    dense4 = tf.layers.dense(dropout4, units=2048, activation=tf.nn.relu, name='dense4')

    logits = tf.layers.dense(dense4, units=params['num_classes'], name='logits')

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
