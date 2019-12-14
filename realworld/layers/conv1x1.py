import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model

tfd = tfp.distributions
tfb = tfp.bijectors


def trainable_lu_factorization(event_size,
                               batch_shape=(),
                               seed=None,
                               dtype=tf.float32,
                               name=None):
    with tf.name_scope('trainable_lu_factorization'):
        event_size = tf.convert_to_tensor(event_size,
                                          dtype=tf.int32,
                                          name='event_size')
        batch_shape = tf.convert_to_tensor(batch_shape,
                                           dtype=event_size.dtype,
                                           name='batch_shape')
        random_matrix = tf.Variable(tf.random.uniform(
            shape=tf.concat([batch_shape, [event_size, event_size]], axis=0),
            dtype=dtype,
            seed=seed,
        ),
                                    name='conv1x1_weights')

        def lu_p(m):
            return tf.linalg.lu(tf.linalg.qr(m).q)

        # lower_upper = tfp.util.DeferredTensor(lambda m: lu_p(m)[0],
        #                                       random_matrix)
        # permutation = tfp.util.DeferredTensor(lambda m: lu_p(m)[1],
        #                                       random_matrix,
        #                                       # trainable=False,
        #                                       dtype=tf.int32,
        #                                       shape=random_matrix.shape[:-1])
        lower_upper = tf.Variable(lu_p(random_matrix)[0], name='lower_upper')
        # ref https://github.com/tensorflow/probability/issues/545
        permutation = tf.Variable(lu_p(random_matrix)[1],
                                  trainable=False,
                                  name='permutation')
        return lower_upper, permutation


def build_model(channels=3):
    # conv1x1 setup
    t_lower_upper, t_permutation = trainable_lu_factorization(channels)
    conv1x1 = tfb.MatvecLU(t_lower_upper, t_permutation, name='MatvecLU')
    print('conv1x1 variable\n', conv1x1.variables)
    inv_conv1x1 = tfb.Invert(conv1x1)

    # forward setup
    fwd = tfp.layers.DistributionLambda(
        lambda x: conv1x1(tfd.Deterministic(x)))
    fwd.vars = conv1x1.trainable_variables

    # inverse setup
    inv = tfp.layers.DistributionLambda(
        lambda x: inv_conv1x1(tfd.Deterministic(x)))
    inv.vars = inv_conv1x1.trainable_variables

    x: tf.Tensor = tf.keras.Input(shape=[28, 28, channels])
    fwd_x: tfp.distributions.TransformedDistribution = fwd(x)
    rev_fwd_x: tfp.distributions.TransformedDistribution = inv(fwd_x)
    example_model = tf.keras.Model(inputs=x, outputs=rev_fwd_x, name='conv1x1')
    return example_model


def test_conv1x1():
    example_model = build_model()
    example_model.trainable = True
    example_model.summary()

    real_x = tf.random.uniform(shape=[2, 28, 28, 3], dtype=tf.float32)
    if example_model.weights == []:
        print('No Trainable Variable exists')
    else:
        print('Some Trainable Variables exist')

    with tf.GradientTape() as tape:
        tape.watch(real_x)
        out_x = example_model(real_x)
        out_x = out_x
        loss = out_x - real_x
    print(tf.math.reduce_sum(real_x - out_x))
    # => nealy 0
    # ex. tf.Tensor(1.3522818e-05, shape=(), dtype=float32)

    try:
        print(tape.gradient(loss, real_x).shape)
    except Exception as e:
        print('Cannot Calculate Gradient')
        print(e)
