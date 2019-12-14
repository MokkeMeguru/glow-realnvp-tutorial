import tensorflow as tf


def load_dataset():
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    SHUFFLE_BUFFER_SIZE = 1000
    BATCH_SIZE = 120

    @tf.function
    def _parse_function(img, label):
        feature = {}
        img = tf.pad(img, paddings=[[2, 2], [2, 2]], mode="CONSTANT")
        img = tf.expand_dims(img, axis=-1)
        img = tf.reshape(img, [32, 32, 1])
        img = tf.cast(img, dtype=tf.float32)
        img = (img / (255.0 / 2)) - 1
        feature["img"] = img
        feature["label"] = label
        return feature

    train_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (train_x, train_y)).map(_parse_function)
    test_dataset_raw = tf.data.Dataset.from_tensor_slices(
        (test_x, test_y)).map(_parse_function)
    train_dataset = train_dataset_raw.shuffle(SHUFFLE_BUFFER_SIZE).batch(
        BATCH_SIZE)
    test_dataset = test_dataset_raw.shuffle(SHUFFLE_BUFFER_SIZE).batch(
        BATCH_SIZE)
    return train_dataset, test_dataset
