# ref https://github.com/tensorflow/probability/issues/545
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd

# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()
# import tensorflow_probability as tfp
# tfb, tfd = tfp.bijectors, tfp.distributions


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
        #                                       dtype=tf.int32,
        #                                       shape=random_matrix.shape[:-1])
        lower_upper = tf.Variable(lu_p(random_matrix)[0])
        # ref https://github.com/tensorflow/probability/issues/545
        permutation = tf.Variable(lu_p(random_matrix)[1], trainable=False)
        return lower_upper, permutation


def build_model(channels=3):
    # conv1x1 setup
    t_lower_upper, t_permutation = trainable_lu_factorization(channels)
    conv1x1 = tfb.MatvecLU(t_lower_upper, t_permutation, name='MatvecLU')
    inv_conv1x1 = tfb.Invert(conv1x1)

    # forward setup
    fwd = tfp.layers.DistributionLambda(
        lambda x: conv1x1(tfd.Deterministic(x)))
    print(conv1x1.trainable_variables)
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


def main():
    print('tensorflow : ', tf.__version__)  # 2.0.0-rc0
    print('tensorflow-probability : ', tfp.__version__)  # 0.8.0-rc0
    # setup environment

    example_model = build_model()
    example_model.trainable = False
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


if __name__ == '__main__':
    main()

##########################################################
# tensorflow :  2.0.0-rc0
# tensorflow-probability :  0.8.0-rc0
# Model: "conv1x1"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_17 (InputLayer)        [(None, 28, 28, 3)]       0
# _________________________________________________________________
# distribution_lambda_31 (Dist ((None, 28, 28, 3), (None 9
# _________________________________________________________________
# distribution_lambda_32 (Dist ((None, 28, 28, 3), (None 9
# =================================================================
# Total params: 9
# Trainable params: 0
# Non-trainable params: 9
# _________________________________________________________________
# Some Trainable Variables exist
# tf.Tensor(-2.0933454e-05, shape=(), dtype=float32)
# (2, 28, 28, 3)
###########################################################
