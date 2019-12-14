import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from layers.conv1x1 import trainable_lu_factorization
from layers.actnorm import Actnorm
from layers.affineCoupling import RealNVP
from layers.squeeze3d import Squeeze3D
from layers.blockwise3d import Blockwise3D

tfb = tfp.bijectors


def gen_flowSteps(
        # for realnvp
        input_shape: list,
        n_hidden: list = [128, 128],
        # for flowStep
        k=4,
        forward_min_event_ndims: int = 3,
        validate_args: bool = False,
        name: str = "flow_step",
):
    flow_step_list = []
    for i in range(k):
        t_lower_upper, t_permutation = trainable_lu_factorization(
            input_shape[-1])
        flow_step_list.append(Actnorm(input_shape[-1]))
        flow_step_list.append(
            tfb.MatvecLU(t_lower_upper,
                         t_permutation,
                         name="{}_{}/matveclu".format(name, i))),
        flow_step_list.append(
            RealNVP(
                input_shape=input_shape,
                n_hidden=n_hidden,
                validate_args=validate_args,
                name="{}_{}/realnvp".format(name, i),
            ))

    flowSteps = tfb.Chain(list(reversed(flow_step_list)),
                          validate_args=validate_args,
                          name=name)
    return flowSteps


def test_gen_flowSteps():
    flowSteps = gen_flowSteps(k=2,
                              input_shape=[16, 16, 4],
                              forward_min_event_ndims=0,
                              name="flowstep_0")
    x = tf.keras.Input([16, 16, 4])
    y = flowSteps(x)
    # tf.keras.Model(x, y).summary()

    x = tf.random.normal([6, 16, 16, 4])
    y = flowSteps.forward(x)
    z = flowSteps.inverse(y)
    return tf.reduce_sum(z - x)


def gen_flow(input_shape, level=3, flow_step_args: dict = None):
    def _gen_input_shapes(input_shape, level):
        input_shape = input_shape
        input_shapes = []
        for i in range(level):
            input_shape = [
                input_shape[0] // 2,
                input_shape[1] // 2,
                input_shape[2] * 2,
            ]
            input_shapes.append(input_shape)
        return input_shapes

    input_shape[-1] = input_shape[-1] * 2
    input_shapes = _gen_input_shapes(input_shape, level)

    def _add_flow(_input_shapes, flow_step_args):
        flow_lists = []
        flow_lists.append(
            Squeeze3D(name="Squeeze_{}".format(level - len(_input_shapes))))
        flowSteps = gen_flowSteps(
            k=2,
            input_shape=_input_shapes[0],
            name="Flowsteps_{}".format(level - len(_input_shapes)),
        )
        flow_lists.append(flowSteps)
        if len(_input_shapes) != 1:
            flow_lists.append(
                Blockwise3D(
                    [
                        tfb.Identity(),
                        tfb.Chain(
                            list(
                                reversed(
                                    _add_flow(_input_shapes[1:],
                                              flow_step_args)))),
                    ],
                    name="Blockwise3D_{}".format(level - len(_input_shapes)),
                ))
        flow_lists.append(
            tfb.Invert(
                Squeeze3D(name="Unsqueeze_{}".format(level -
                                                     len(_input_shapes)))))
        return flow_lists

    return tfb.Chain(list(reversed(_add_flow(input_shapes, level))))


def test_gen_flow():
    flow = gen_flow([32, 32, 1])
    print(len(flow.trainable_variables))
    x = tf.keras.Input([32, 32, 1])
    y = flow.forward(x)
    # tf.keras.Model(x, y).summary()
    tf.keras.utils.plot_model(tf.keras.Model(x, y),
                              show_shapes=True,
                              to_file="realnvp.png")
    x = tf.random.normal([3, 32, 32, 1])
    y = flow.forward(x)
    z = flow.inverse(y)
    return tf.reduce_sum(z - x)
