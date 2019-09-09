import os
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Glow import Glow
import tensorflow_probability.python.bijectors as tfb


def parse_function(serialized_example):
    feature = tf.io.parse_single_example(serialized_example,
                                         features={
                                             'data':
                                             tf.io.FixedLenFeature(
                                                 (),
                                                 dtype=tf.string,
                                                 default_value=''),
                                             'label':
                                             tf.io.FixedLenFeature(
                                                 (),
                                                 dtype=tf.int64,
                                                 default_value=0)
                                         })
    data = tf.io.decode_raw(feature['data'], out_type=tf.uint8)
    data = tf.reshape(data, [28, 28, 1])
    data = tf.pad(data, paddings=[[2, 2], [2, 2], [0, 0]], mode='CONSTANT')
    data = tf.cast(data, dtype=tf.float32)
    data = (data / (255.0 * 2)) - 1
    data = tf.image.grayscale_to_rgb(data)
    # image shape: [32, 32, 3]
    label = tf.cast(feature['label'], dtype=tf.int32)
    return data, label


def main():
    trainset = tf.data.TFRecordDataset(
        os.path.join(
            'mnists',
            'trainset.tfrecord')).map(parse_function).shuffle(100).batch(20)
    testset = tf.data.TFRecordDataset(
        os.path.join('mnists',
                     'testset.tfrecord')).map(parse_function).batch(100)

    # Create a model
    base_dist = tfp.distributions.Normal(loc=0., scale=1.)
    glow = tfb.Invert(Glow([None, 32, 32, 3], levels=2))

    optimizer = tf.keras.optimizers.Adam(1e-4)

    # train checkpoint
    # checkpoint = tf.train.Checkpoint(model=glow,
    #                                optimizer=optimizer,
    #                               optimizer_step=optimizer.iterations)
    # checkpoint.restore(tf.train.latest_checkpoint('checkpoints'))
    # log = tf.summary.create_file_writer('checkpoints')
    print('training...')

    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    while True:
        for (images, _) in trainset:
            with tf.GradientTape() as tape:
                code = glow.inverse(images)
                loss = -tf.reduce_mean(base_dist.log_prob(code))
            avg_loss.update_state(loss)

            # write log
            if tf.equal(optimizer.iterations % 100, 0):
                print(loss)
                #     with log.as_default():
                #         tf.summary.scalar('loss',
                #                           avg_loss.result(),
                #                           step=optimizer.iterations)
                #         sample_image = glow.forward(code[0:1])
                #     tf.summary.image(
                #         'samgple',
                #         base_dist.sample((1, 8, 8, 24)),  # sample_image,
                #         step=optimizer.iterations)
                print('Step {} :  Loss {:.6f}'.format(
                    optimizer.iterations.numpy(), avg_loss.result()))
                avg_loss.reset_states()

            grads = tape.gradient(loss, glow.trainable_variables)
            optimizer.apply_gradients(zip(grads, glow.trainable_variables))

        # checkpoint.save(os.path.join('checkpoints', 'ckpt'))
        if loss < 0.5: break

    if not os.path.exists('model'):
        os.mkdir('model')
    glow.save_weights('./model/glow_model')


def _test():
    avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
    trainset = tf.data.TFRecordDataset(
        os.path.join(
            'mnists',
            'trainset.tfrecord')).map(parse_function).shuffle(100).batch(100)

    base_dist = tfp.distributions.Normal(loc=0., scale=1.)
    glow = tfb.Invert(Glow([None, 32, 32, 3], levels=2))
    optimizer = tf.keras.optimizers.Adam(1e-6)

    for batch in trainset.take(100):
        train_example = batch[0]

        with tf.GradientTape() as tape:
            code = glow.inverse(train_example)
            loss = -tf.reduce_mean(base_dist.log_prob(code))
            avg_loss.update_state(loss)
            # print(avg_loss.result())

            grads = tape.gradient(loss, glow.trainable_variables)
            optimizer.apply_gradients(zip(grads, glow.trainable_variables))

        with tf.GradientTape() as tape:
            code = glow.inverse(train_example)
            loss = -tf.reduce_mean(base_dist.log_prob(code))
            avg_loss.update_state(loss)
            # print(avg_loss.result())
        if tf.equal(optimizer.iterations % 10, 0):
            print('Step {} :  Loss {:.6f}'.format(optimizer.iterations.numpy(),
                                                  avg_loss.result()))
            avg_loss.reset_states()


# In [835]: _test()
# batch_shape 12
# batch_shape 12
# batch_shape 24
# batch_shape 24
# Step 10 :  Loss 1.478505
# Step 20 :  Loss 1.425855
# Step 30 :  Loss 1.390535
# Step 40 :  Loss 1.348447
# Step 50 :  Loss 1.322992
# Step 60 :  Loss 1.299943
# Step 70 :  Loss 1.268664
# Step 80 :  Loss 1.246166
# Step 90 :  Loss 1.231701
# Step 100 :  Loss 1.217493

if __name__ == '__main__':
    main()

#############################
# Limit:                  6732647629
# InUse:                  6730854912
# MaxInUse:               6732391936
# NumAllocs:                  727103
# MaxAllocSize:             46268416

#
# 2019-09-09 10:08:50.759092: W tensorflow/core/common_runtime/bfc_allocator.cc:424] ****************************************************************************************************
# 2019-09-09 10:08:50.759121: W tensorflow/core/framework/op_kernel.cc:1622] OP_REQUIRES failed at conv_grad_input_ops.cc:415 : Resource exhausted: OOM when allocating tensor with shape[100,16,16,512] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
# Traceback (most recent call last):
#   File "train_eager.py", line 70, in <module>
#     main();
#   File "train_eager.py", line 57, in main
#     grads = tape.gradient(loss, glow.trainable_variables);
#   File "/home/meguru/Github/glow-flow/venv/lib/python3.7/site-packages/tensorflow_core/python/eager/backprop.py", line 1014, in gradient
#     unconnected_gradients=unconnected_gradients)
#   File "/home/meguru/Github/glow-flow/venv/lib/python3.7/site-packages/tensorflow_core/python/eager/imperative_grad.py", line 76, in imperative_grad
#     compat.as_str(unconnected_gradients.value))
#   File "/home/meguru/Github/glow-flow/venv/lib/python3.7/site-packages/tensorflow_core/python/eager/backprop.py", line 138, in _gradient_function
#     return grad_fn(mock_op, *out_grads)
#   File "/home/meguru/Github/glow-flow/venv/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_grad.py", line 596, in _Conv2DGrad
#     data_format=data_format),
#   File "/home/meguru/Github/glow-flow/venv/lib/python3.7/site-packages/tensorflow_core/python/ops/gen_nn_ops.py", line 1372, in conv2d_backprop_input
#     _six.raise_from(_core._status_to_exception(e.code, message), None)
#   File "<string>", line 3, in raise_from
# tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[100,16,16,512] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:Conv2DBackpropInput]

# TODO: via terminal, It's training may be failed ....
#  多分これはフレームワークの問題このあたりの issue を書く
# TODO: loss が下がらない
# 多分これはライブラリの問題・学習するパラメータを おそらく認識できていない
# ref tensorflow/tensorflow-probability #545
#
# main()
# training...
# tf.Tensor(2.634636, shape=(), dtype=float32)
# Step 0 :  Loss 2.634636
# tf.Tensor(0.9256006, shape=(), dtype=float32)
# Step 100 :  Loss 1.060871
# tf.Tensor(0.9204561, shape=(), dtype=float32)
# Step 200 :  Loss 0.922022
# tf.Tensor(0.91958785, shape=(), dtype=float32)
# Step 300 :  Loss 0.919873
# tf.Tensor(0.91927886, shape=(), dtype=float32)
# Step 400 :  Loss 0.919416
# tf.Tensor(0.9191531, shape=(), dtype=float32)
# Step 500 :  Loss 0.919224
# tf.Tensor(0.91912687, shape=(), dtype=float32)
# Step 600 :  Loss 0.919141
# tf.Tensor(0.91909295, shape=(), dtype=float32)
# Step 700 :  Loss 0.919101
# tf.Tensor(0.91904324, shape=(), dtype=float32)
# Step 800 :  Loss 0.919065
# tf.Tensor(0.919039, shape=(), dtype=float32)
# Step 900 :  Loss 0.919043
# tf.Tensor(0.9190367, shape=(), dtype=float32)
# Step 1000 :  Loss 0.919031
# tf.Tensor(0.9190061, shape=(), dtype=float32)
# Step 1100 :  Loss 0.919018
# tf.Tensor(0.91901004, shape=(), dtype=float32)
# Step 1200 :  Loss 0.919010
# tf.Tensor(0.91899383, shape=(), dtype=float32)
# Step 1300 :  Loss 0.919003
# tf.Tensor(0.9189888, shape=(), dtype=float32)
# Step 1400 :  Loss 0.918998
# tf.Tensor(0.91899323, shape=(), dtype=float32)
# Step 1500 :  Loss 0.918999
# tf.Tensor(0.9189841, shape=(), dtype=float32)
# Step 1600 :  Loss 0.918992
# tf.Tensor(0.91897774, shape=(), dtype=float32)
# Step 1700 :  Loss 0.918988
# tf.Tensor(0.9189818, shape=(), dtype=float32)
# Step 1800 :  Loss 0.918991
# tf.Tensor(0.9230054, shape=(), dtype=float32)
# Step 1900 :  Loss 0.924658
# tf.Tensor(0.9190056, shape=(), dtype=float32)
# Step 2000 :  Loss 0.919281
# tf.Tensor(0.91899455, shape=(), dtype=float32)
# Step 2100 :  Loss 0.918995
# tf.Tensor(0.9189751, shape=(), dtype=float32)
# Step 2200 :  Loss 0.918985
# tf.Tensor(0.9189748, shape=(), dtype=float32)
# Step 2300 :  Loss 0.918979
# tf.Tensor(0.9189808, shape=(), dtype=float32)
# Step 2400 :  Loss 0.918977
# tf.Tensor(0.9189701, shape=(), dtype=float32)
# Step 2500 :  Loss 0.918975
# tf.Tensor(0.9189719, shape=(), dtype=float32)
# Step 2600 :  Loss 0.918972
# tf.Tensor(0.9189726, shape=(), dtype=float32)
# Step 2700 :  Loss 0.918969
# tf.Tensor(0.91897076, shape=(), dtype=float32)
# Step 2800 :  Loss 0.918969
# tf.Tensor(0.91896725, shape=(), dtype=float32)
# Step 2900 :  Loss 0.918968
# tf.Tensor(0.9189659, shape=(), dtype=float32)
# Step 3000 :  Loss 0.918967
# tf.Tensor(0.9189653, shape=(), dtype=float32)
# Step 3100 :  Loss 0.918966
# tf.Tensor(0.91896284, shape=(), dtype=float32)
# Step 3200 :  Loss 0.918964
# tf.Tensor(0.9189665, shape=(), dtype=float32)
# Step 3300 :  Loss 0.918964
# tf.Tensor(0.91896605, shape=(), dtype=float32)
# Step 3400 :  Loss 0.918962
# tf.Tensor(0.91896015, shape=(), dtype=float32)
# Step 3500 :  Loss 0.918962
# tf.Tensor(0.9189623, shape=(), dtype=float32)
# Step 3600 :  Loss 0.918962
# tf.Tensor(0.9189602, shape=(), dtype=float32)
# Step 3700 :  Loss 0.918961
# tf.Tensor(0.9189611, shape=(), dtype=float32)
# Step 3800 :  Loss 0.918962
# tf.Tensor(0.91898614, shape=(), dtype=float32)
# Step 3900 :  Loss 0.918962
# tf.Tensor(0.9189654, shape=(), dtype=float32)
# Step 4000 :  Loss 0.918964
# tf.Tensor(0.91898674, shape=(), dtype=float32)
# Step 4100 :  Loss 0.918967
# tf.Tensor(0.9191143, shape=(), dtype=float32)
# Step 4200 :  Loss 0.918971
# tf.Tensor(0.9189589, shape=(), dtype=float32)
# Step 4300 :  Loss 0.918965
# tf.Tensor(0.9189588, shape=(), dtype=float32)
# Step 4400 :  Loss 0.918975
# tf.Tensor(0.91899645, shape=(), dtype=float32)
# Step 4500 :  Loss 0.918983
# tf.Tensor(0.91896373, shape=(), dtype=float32)
# Step 4600 :  Loss 0.918959
# tf.Tensor(0.91896635, shape=(), dtype=float32)
# Step 4700 :  Loss 0.918966
# tf.Tensor(0.91958547, shape=(), dtype=float32)
# Step 4800 :  Loss 0.919021
# tf.Tensor(0.91895497, shape=(), dtype=float32)
# Step 4900 :  Loss 0.918985
