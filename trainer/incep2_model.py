import tensorflow as tf


def model_fn(features, labels, mode, params):
    x = tf.reshape(features, [-1, 99, 161, 1], name='input_incep2s')
    x_norm = tf.layers.batch_normalization(x, training=mode == tf.estimator.ModeKeys.TRAIN, name='x_norm')
    if params['verbose_summary']:
        tf.summary.image('input', x)

    incep1 = inception_block(x_norm, name='incep1')
    incep2 = inception_block(incep1, t1x1=16, t3x3=16, t5x5=16, tmp=16, name='incep2')
    incep3 = inception_block(incep2, t1x1=32, t3x3=32, t5x5=32, tmp=32, name='incep3')
    incep4 = inception_block(incep3, t1x1=64, t3x3=64, t5x5=64, tmp=64, name='incep4')

    dim = incep4.get_shape()[1:]
    dim = int(dim[0] * dim[1] * dim[2])
    flat = tf.reshape(incep4, [-1, dim], name='flat')
    dropout5 = tf.layers.dropout(flat, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN, name='dropout5')
    dense5 = tf.layers.dense(dropout5, units=2048, activation=tf.nn.relu, name='dense5')

    logits = tf.layers.dense(dense5, units=params['num_classes'], name='logits')

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


def inception_block(prev, t1x1=8, t3x3=8, t5x5=8, tmp=8, name='incep'):
    with tf.variable_scope(name):
        with tf.variable_scope('1x1_conv'):
            tower_1x1 = tf.layers.conv2d(prev,
                                         filters=t1x1,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')

        with tf.variable_scope('3x3_conv'):
            tower_3x3 = tf.layers.conv2d(prev,
                                         filters=t3x3,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
            tower_3x3 = tf.layers.conv2d(tower_3x3,
                                         filters=t3x3,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv')

        with tf.variable_scope('5x5_conv'):
            tower_5x5 = tf.layers.conv2d(prev,
                                         filters=t5x5,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
            tower_5x5 = tf.layers.conv2d(tower_5x5,
                                         filters=t5x5,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv_1')
            tower_5x5 = tf.layers.conv2d(tower_5x5,
                                         filters=t5x5,
                                         kernel_size=3,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='3x3_conv_2')

        with tf.variable_scope('maxpool'):
            tower_mp = tf.layers.max_pooling2d(prev,
                                               pool_size=3,
                                               strides=1,
                                               padding='same',
                                               name='3x3_maxpool')
            tower_mp = tf.layers.conv2d(tower_mp,
                                         filters=tmp,
                                         kernel_size=1,
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='1x1_conv')
        return tf.concat([tower_1x1, tower_3x3, tower_5x5, tower_mp], axis=3)
