import os
import numpy as np
import tensorflow as tf
import pickle


def write_tfrecord(data, label, filename):
    assert data.shape[1:] == (28, 28), 'image shape is invalid'
    if os.path.exists(filename):
        os.remove(filename)
    label = np.array(label, dtype=np.int64)
    writer = tf.io.TFRecordWriter(filename)
    for i in range(data.shape[0]):
        trainsample = tf.train.Example(features=tf.train.Features(
            feature={
                'data':
                tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[data[i, ...].tobytes()])),
                'label':
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[label[i]]))
            }))
        writer.write(trainsample.SerializeToString())
    writer.close()


def create_mnist():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    # train data size 60000
    if not os.path.exists('mnists'):
        os.mkdir('mnists')
    write_tfrecord(train_x, train_y,
                   os.path.join('mnists', 'trainset.tfrecord'))
    write_tfrecord(test_x, test_y, os.path.join('mnists', 'testset.tfrecord'))


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
    # data = tf.image.grayscale_to_rgb(data)
    # image shape: [32, 32, 3]
    # image shape: [32, 32, 1]
    label = tf.cast(feature['label'], dtype=tf.int32)
    return data, label


if __name__ == '__main__':
    create_mnist()
