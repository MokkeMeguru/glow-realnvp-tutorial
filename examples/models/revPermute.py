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
"""Builds a RevPermute Bijector
ref. jupyter_notebooks/RealNVP_for_mnist .ipynb
origin. https://github.com/MokkeMeguru/glow-realnvp-tutorial
test. tensorflow==2.0.0rc0, tensorflow-probability==0.8.0rc0
"""

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class RevPermute(tfb.Bijector):
    def __init__(
            self,
            axis=[-1],
            forward_min_event_ndims=0,
            validate_args=False,
            name="RevPermute",
    ):
        super(RevPermute,
              self).__init__(forward_min_event_ndims=forward_min_event_ndims,
                             validate_args=validate_args,
                             name=name,
                             is_constant_jacobian=True)
        self._axis = axis

    @property
    def axis(self):
        return self._axis

    def _forward(self, x):
        return tf.reverse(x, self.axis)

    def _inverse(self, y):
        return tf.reverse(y, self.axis)

    def _forward_log_det_jacobian(self, x):
        return tf.constant(0.0, dtype=x.dtype)

    def _inverse_log_det_jacobian(self, y):
        return tf.constant(0.0, dtype=y.dtype)


def test_revPermute():
    target_dist = tfd.Normal(loc=0., scale=1.)
    revPermute = RevPermute()
    x = tf.keras.Input([16, 16, 4])
    y = revPermute.forward(x)
    # tf.keras.Model(x, y).summary()

    x = tf.random.normal([2, 16, 16, 4])
    y = revPermute.forward(x)
    z = revPermute.inverse(y)
    flow = tfd.TransformedDistribution(event_shape=[16, 16, 4],
                                       distribution=target_dist,
                                       bijector=revPermute)
    print(tf.reduce_mean(flow.log_prob(tf.random.normal([2, 16, 16, 4]))))
    return tf.reduce_sum(z - x)
