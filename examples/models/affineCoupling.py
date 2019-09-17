# Copyright 2019 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Builds a RealNVP(for 3D) Bijector
ref. jupyter_notebooks/RealNVP_for_mnist .ipynb
origin. https://github.com/MokkeMeguru/glow-realnvp-tutorial
test. tensorflow==2.0.0rc0, tensorflow-probability==0.8.0rc0
"""

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer, Conv2D
from tensorflow.keras import Model

tfd = tfp.distributions
tfb = tfp.bijectors


class NN(Layer):
    def __init__(self,
                 input_shape,
                 n_hidden=[512, 512],
                 kernel_size=[[3, 3], [1, 1]],
                 strides=[[1, 1], [1, 1]],
                 activation="relu",
                 name=None):
        if name:
            super(NN, self).__init__(name=name)
        else:
            super(NN, self).__init__()
        layer_list = []
        for i, (hidden, kernel,
                stride) in enumerate(zip(n_hidden, kernel_size, strides)):
            layer_list.append(
                Conv2D(
                    hidden,
                    kernel_size=kernel,
                    strides=stride,
                    activation=activation,
                    padding='SAME',
                    name="dense_{}_1".format(i),
                ))
        self.layer_list = layer_list
        self.log_s_layer = Conv2D(
            input_shape,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            kernel_initializer="zeros",
            activation="tanh",
            name="log_s",
        )
        self.t_layer = Conv2D(
            input_shape,
            kernel_size=[3, 3],
            strides=[1, 1],
            padding='SAME',
            kernel_initializer="zeros",
            name="t",
        )

    def call(self, x):
        y = x
        for layer in self.layer_list:
            y = layer(y)
        log_s = self.log_s_layer(y)
        t = self.t_layer(y)
        return log_s, t


def nn_test():
    nn = NN(2, [256, 256])
    x = tf.keras.Input([16, 16, 2])
    log_s, t = nn(x)
    # Non trainable params: -> Batch Normalization's params
    tf.keras.Model(x, [log_s, t], name="nn_test").summary()


class RealNVP(tfb.Bijector):
    def __init__(
            self,
            input_shape,
            forward_min_event_ndims=3,
            validate_args: bool = False,
            name="real_nvp",
            n_hidden=[512, 512],
            **kargs,
    ):
        """
        Args:
            input_shape:
                input_shape,
                ex. [28, 28, 3] (image) [2] (x-y vector)
            forward_min_event_ndims:
                this bijector do
                1. element-wize quantities => 0
                2. vector-wize quantities => 1
                3. matrix-wize quantities => 2
                4. tensor-wize quantities => 3
            n_hidden:
                see. class NN
            **kargs:
                see. class NN
                you can inuput NN's layers parameter here.
        """
        super(RealNVP, self).__init__(
            validate_args=validate_args,
            forward_min_event_ndims=forward_min_event_ndims,
            name=name,
        )

        assert input_shape[-1] % 2 == 0
        self.input_shape = input_shape
        nn_layer = NN(
            input_shape[-1] // 2,
            n_hidden=n_hidden,
        )
        nn_input_shape = input_shape.copy()
        nn_input_shape[-1] = input_shape[-1] // 2
        x = tf.keras.Input(nn_input_shape)
        log_s, t = nn_layer(x)
        self.nn = Model(x, [log_s, t], name=self.name + "/nn")

    def _forward(self, x):
        x_a, x_b = tf.split(x, 2, axis=-1)
        y_b = x_b
        log_s, t = self.nn(x_b)
        s = tf.exp(log_s)
        y_a = s * x_a + t
        y = tf.concat([y_a, y_b], axis=-1)
        return y

    def _inverse(self, y):
        y_a, y_b = tf.split(y, 2, axis=-1)
        x_b = y_b
        log_s, t = self.nn(y_b)
        s = tf.exp(log_s)
        x_a = (y_a - t) / s
        x = tf.concat([x_a, x_b], axis=-1)
        return x

    def _forward_log_det_jacobian(self, x):
        _, x_b = tf.split(x, 2, axis=-1)
        log_s, t = self.nn(x_b)
        return tf.reduce_sum(log_s)


def realnvp_test():
    realnvp = RealNVP(input_shape=[16, 16, 4], n_hidden=[256, 256])
    x = tf.keras.Input([16, 16, 4])
    y = realnvp.forward(x)
    print("trainable_variables :", len(realnvp.trainable_variables))

    flow = tfd.TransformedDistribution(
        event_shape=[16, 16, 4],
        distribution=tfd.Normal(loc=0.0, scale=1.0),
        bijector=realnvp,
    )
    x = flow.sample(64)
    y = realnvp.inverse(x)
    log_prob = flow.log_prob(y)
    print(
        x.shape,
        y.shape,
        log_prob.shape,
        # -tf.reduce_mean(log_prob),
        # -tf.reduce_mean(flow.distribution.log_prob(x)),
        # -tf.reduce_mean(
        #     flow.bijector.forward_log_det_jacobian(
        #         x, event_ndims=flow._maybe_get_static_event_ndims()
        #     )
        # ),
        # -tf.reduce_mean(flow._log_prob(x)),
        # flow._finish_log_prob_for_one_fiber(
        #     y,
        #     x,
        #     flow.bijector.forward_log_det_jacobian(
        #         x, event_ndims=flow._maybe_get_static_event_ndims()
        #     ),
        #     flow._maybe_get_static_event_ndims(),
        #
        # ),
        # tf.reduce_sum(flow.distribution.log_prob(
        #     flow._maybe_rotate_dims(x, rotate_right=True)),
        #               axis=flow._reduce_event_indices)
    )
