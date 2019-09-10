import tensorflow_probability as tfp
import tensorflow_probability.python.bijectors as tfb
from bijectors import conv1x1_official, actNorm, affineCoupling
import numpy as np


class GlowStep(tfp.bijectors.Bijector):
    def __init__(self,
                 batch_shape,
                 depth=2,
                 validate_args=False,
                 name='Glow_step'):
        super(GlowStep, self).__init__(forward_min_event_ndims=3,
                                       validate_args=validate_args,
                                       name=name)
        layers = []
        batch_shape = np.array(batch_shape)
        for i in range(depth):
            print(i, ':', batch_shape[-1])
            t_lower_upper, t_permutation = conv1x1_official.trainable_lu_factorization(
                batch_shape[-1])
            layers.append(
                actNorm.ActNorm(name=self._name + '/actnorm_{}'.format(i)))
            layers.append(
                tfb.MatvecLU(t_lower_upper,
                             t_permutation,
                             name=self._name + '/matvecLU_{}'.format(i)))
            layers.append(
                affineCoupling.AffineCoupling(batch_shape=batch_shape,
                                              hidden_filters=256,
                                              name=self._name +
                                              '/affinecoupling_{}'.format(i)))
        self.flow = tfp.bijectors.Chain(list(reversed(layers)))

    def _forward(self, x):
        return self.flow.forward(x)

    def _inverse(self, y):
        return self.flow.inverse(y)

    def _inverse_log_det_jacobian(self, y):
        ildj = self.flow.inverse_log_det_jacobian(y, event_ndims=3)
        return ildj
