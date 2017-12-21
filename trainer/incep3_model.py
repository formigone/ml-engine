import tensorflow as tf

from util import inception_block, flatten, double_inception, conv_group



def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep3')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    incep1 = double_inception(x_norm, block_depth=2, name='incep1')
    conv2 = conv_group(incep1, 16, 'conv_group_2', verbose=params['verbose_summary'])
    conv3 = conv_group(conv2, 32, 'conv_group_3', verbose=params['verbose_summary'])

    incep3 = double_inception(conv3, block_depth=8, name='incep3')
    conv4 = conv_group(incep3, 64, 'conv_group_4', verbose=params['verbose_summary'])
    conv5 = conv_group(conv4, 128, 'conv_group_5', verbose=params['verbose_summary'])

    incep6 = double_inception(conv5, block_depth=32, name='incep6')
    incep7 = double_inception(incep6, block_depth=64, name='incep7')
    incep8 = double_inception(incep7, block_depth=128, name='incep8')

    flat = flatten(incep8)
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
