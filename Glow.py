import numpy as np
import tensorflow_probability as tfp
from GlowStep import GlowStep
from bijectors.split import Split
from bijectors.squeeze import Squeeze
import tensorflow as tf
import tensorflow_probability.python.bijectors as tfb
import tensorflow_probability.python.distributions as tfd


class Glow(tfp.bijectors.Bijector):
    def __init__(self,
                 batch_shape,
                 levels=2,
                 depth=2,
                 validate_args=False,
                 name='Glow'):
        super(Glow, self).__init__(forward_min_event_ndims=3,
                                   validate_args=validate_args,
                                   name=name)
        batch_shape = np.array(batch_shape)
        batch_shape = np.array([None, 32, 32, 3])
        layers = []
        for i in range(levels):
            batch_shape[-1] = batch_shape[-1] * 4
            batch_shape[-3:-1] = batch_shape[-3:-1] // 2
            layers.append(
                Squeeze(factor=2,
                        name=self._name + '/space2batch_{}'.format(i)))
            layers.append(
                GlowStep(batch_shape,
                         depth=depth,
                         name=self._name + '/glowstep_{}'.format(i)))
            if i < levels - 1:
                layers.append(
                    Split(batch_shape,
                          name=self._name + '/split_{}'.format(i)))
                batch_shape[-1] = batch_shape[-1] // 2
        self.flow = tfp.bijectors.Chain(list(reversed(layers)))

    def _forward(self, x):
        return self.flow.forward(x)

    def _inverse(self, y):
        return self.flow.inverse(y)

    def _inverse_log_det_jacobian(self, y):
        ildj = self.flow.inverse_log_det_jacobian(y, event_ndims=3)
        return ildj


def main():
    glow = tfb.Invert(Glow([None, 32, 32, 3]))
    x = tf.random.normal([2, 32, 32, 3])
    y = glow.inverse(x)
    z = glow.forward(y)
    print(tf.reduce_mean(x - z))

    # glow = Glow([None, 32, 32, 3])
    # x = tf.random.normal([2, 32, 32, 3])
    # y = glow.forward(x)
    # z = glow.inverse(y)
    # print(tf.reduce_mean(x - z))
