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
from args import args
from model import gen_flow
from dataset import load_dataset
from pathlib import Path
from functools import reduce

tfb = tfp.bijectors
tfd = tfp.distributions
optimizers = tf.keras.optimizers


class Flow_trainer:
    def __init__(self, args=args, training=True):
        self.args = args
        self.flow = tfd.TransformedDistribution(
            event_shape=args['input_shape'],
            distribution=tfd.Normal(0.0, 1.0),
            bijector=gen_flow(args['input_shape'], level=args['level']))
        self.optimizer = optimizers.Adam(learning_rate=1e-5)
        self.loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
        self.train_dataset, self.test_dataset = load_dataset()
        self.epochs = args['epochs']
        self.input_volume = reduce(lambda x, y: x * y, args['input_shape'])
        for sample in self.train_dataset.take(1):
            print('init loss (log_prob) {}'.format(self.calc_loss(sample)))
        self.setup_checkpoint()

    def setup_checkpoint(self,
                         checkpoint_path=Path('./checkpoints/flow_train')):
        ckpt = tf.train.Checkpoint(model=self.flow, optimizer=self.optimizer)
        ckpt_manager = tf.train.CheckpointManager(ckpt,
                                                  checkpoint_path,
                                                  max_to_keep=3)
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('[Flow] Latest checkpoint restored!!')
        self.ckpt = ckpt
        self.ckpt_manager = ckpt_manager
        for sample in self.train_dataset.take(1):
            print('restored loss (log_prob) {}'.format(self.calc_loss(sample)))


    def calc_loss(self, batch):
        print(batch['img'].shape)
        
        def loss(batch):
            return - tf.reduce_mean(self.flow.log_prob(batch['img']))

        return loss(batch)

    def train(self):
        flag = False

        @tf.function
        def loss(target):
            return -tf.reduce_mean(self.flow.log_prob(target['img'])) / (np.log(2.) * self.input_volume)

        for epoch in range(self.epochs):
            if flag:
                print('raise NaN')
                break
            for targets in self.train_dataset:
                with tf.GradientTape() as tape:
                    log_prob_loss = loss(targets)
                grads = tape.gradient(log_prob_loss,
                                      self.flow.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.flow.trainable_variables))
                if tf.math.is_nan(log_prob_loss):
                    flag = True
                    break
                self.loss.update_state(log_prob_loss)
                if tf.equal(self.optimizer.iterations % 1000, 0):
                    print("Step {} Loss {:.6f}".format(
                        self.optimizer.iterations, self.loss.result()))
                if tf.equal(self.optimizer.iterations % 100, 0):
                    # with log.as_default():
                    #     tf.summary.scalar("loss", avg_loss.result(), step=optimizer.iterations)
                    #     avg_loss.reset_states()
                    self.loss.reset_states()
            # save per 1 epoch
            ckpt_save_path = self.ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(
                epoch + 1, ckpt_save_path))

# for training
# =>
# [1]: flowmodel = Flow_trainer()
# ;; init loss (log_prob) 48627494912.0
# [2]: flowmodel.train()
# ;;  wait about an hour or moment...


# for restore
# =>
# [1]: flowmodel = Flow_trainer()
# ;; init loss (log_prob)  193615855616.0
# ;; restored loss (log_prob) 1767812.0


