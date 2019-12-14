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

tfd = tfp.distributions
tfb = tfp.bijectors


class Squeeze3D(tfb.Bijector):
    def __init__(
            self,
            factor=2,
            forward_min_event_ndims=0,
            inverse_min_event_ndims=0,
            validate_args=False,
            name="Squeeze",
    ):
        self._factor = factor
        super(Squeeze3D,
              self).__init__(forward_min_event_ndims=forward_min_event_ndims,
                             inverse_min_event_ndims=inverse_min_event_ndims,
                             name=name,
                             is_constant_jacobian=True)

    @property
    def factor(self):
        return self._factor

    def _forward(self, x):
        (H, W, C) = x.shape[1:]
        batch_size = tf.shape(x)[0:1]
        tmp_shape = tf.concat(
            [
                batch_size,
                (H // self.factor, self.factor, W // self.factor, self.factor,
                 C),
            ],
            axis=0,
        )
        output_shape = tf.concat(
            [
                batch_size,
                (H // self.factor, W // self.factor, C * self.factor**2)
            ],
            axis=0,
        )
        y = tf.reshape(x, tmp_shape)
        y = tf.transpose(y, [0, 1, 3, 5, 2, 4])
        y = tf.reshape(y, output_shape)
        return y

    def _inverse(self, y):
        (H, W, C) = y.shape[1:]
        batch_size = tf.shape(y)[0:1]
        tmp_shape = tf.concat([
            batch_size, (H, W, C // self.factor**2, self.factor, self.factor)
        ],
                              axis=0)
        output_shape = tf.concat([
            batch_size, (H * self.factor, W * self.factor, C // self.factor**2)
        ],
                                 axis=0)
        x = tf.reshape(y, tmp_shape)
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, output_shape)
        return x

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0.0, dtype=x.dtype)


def test_squeeze3D():
    factor = 2
    x = tf.Variable([[[1, 2, 5, 6], [3, 4, 7, 8], [9, 10, 13, 14],
                      [11, 12, 15, 16]]])
    x = tf.expand_dims(x, axis=-1)
    squeeze3d = Squeeze3D(factor=factor)
    y = squeeze3d.forward(x)
    z = squeeze3d.inverse(y)
    print(tf.reduce_sum(x - z))

    flow = tfd.TransformedDistribution(event_shape=[16, 16, 2],
                                       distribution=tfd.Normal(loc=0.,
                                                               scale=1.),
                                       bijector=squeeze3d)
    x = tf.random.normal([64, 16, 16, 2])
    y = flow.bijector.forward(x)
    log_prob = flow.log_prob(y)
    print(x.shape, y.shape, log_prob.shape)
