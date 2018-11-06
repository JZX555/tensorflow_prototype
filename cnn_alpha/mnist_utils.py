import os
import struct

import numpy as np
import tensorflow as tf

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MNIST_PATH = "/Users/barid/Documents/batch_train_data/image-MNIST"


def parse_data(path, dataset, flatten):
    if dataset != 'train' and dataset != 't10k':
        raise NameError('dataset must be train or t10k')

    label_file = os.path.join(path, dataset + '-labels-idx1-ubyte')
    with open(label_file, 'rb') as file:
        _, num = struct.unpack(">II", file.read(8))
        labels = np.fromfile(file, dtype=np.int8)  # int8
        new_labels = np.zeros((num, 10))
        new_labels[np.arange(num), labels] = 1

    img_file = os.path.join(path, dataset + '-images-idx3-ubyte')
    with open(img_file, 'rb') as file:
        _, num, rows, cols = struct.unpack(">IIII", file.read(16))
        imgs = np.fromfile(
            file, dtype=np.uint8).reshape(num, rows, cols)  # uint8
        imgs = imgs.astype(np.float32) / 255.0
        if flatten:
            imgs = imgs.reshape([num, -1])

    return imgs, new_labels


def read_mnist(path, flatten=True, num_train=5500):
    """
    Read in the mnist dataset, given that the data is stored in path
    Return
    ((train_imgs, train_labels), (val_img, val_labels),(test_imgs, test_labels))
    """
    imgs, labels = parse_data(path, 'train', flatten)
    indices = np.random.permutation(labels.shape[0])
    train_idx, val_idx = indices[:num_train], indices[num_train:]
    train_img, train_labels = imgs[train_idx, :], labels[train_idx, :]
    val_img, val_labels = imgs[val_idx, :], labels[val_idx, :]
    test_imgs, test_labels = parse_data(path, 't10k', flatten)
    return (train_img, train_labels), (val_img, val_labels), (test_imgs, test_labels)


def get_mnist_dataset_local(batch_size, local_path):
    # Step 1: Read in data
    mnist_folder = local_path
    train, val, test = read_mnist(mnist_folder, flatten=False)
    # Step 2: Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train)  # slice! slice! slice!
    train_data = train_data.shuffle(10000)  # if you want to shuffle your data
    train_data = train_data.batch(batch_size)

    val_data = tf.data.Dataset.from_tensor_slices(val)
    val_data = val_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices(test)
    test_data = test_data.batch(batch_size)

    return train_data, val_data, test_data
