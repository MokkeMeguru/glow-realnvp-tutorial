import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras import layers
import tensorflow_probability.python.bijectors as tfb


def _use_static_shape(input_tensor, ndims):
    return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


class Parallel(tfb.Bijector):
    """Bijector which applies a set of bijectors in parallel"""
    
