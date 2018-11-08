# encoding=utf8

import tensorflow as tf
import numpy as np
import os
# DATA_PATH = '/Users/barid/Documents/batch_train_data/corpus-ptb/data/'

TOKEN = '<eos>'


def read_data(file_name, local=True):
    """
    """
    if local:
        with open(file_name, 'r', encoding='utf8') as f:
            for k, v in enumerate(f):
                yield v.replace("\n", TOKEN).split()
    else:
        with tf.gfile.GFile(name=file_name) as f:
            return f.read().replace("\n", TOKEN).split()
    pass


def fix_sequence_length(time_step, sentence):
    length = len(sentence)
    if length >= time_step:
        sentence = sentence[:time_step]
    else:
        for i in range(length, time_step):
            sentence = np.concatenate((np.zeros((1)), sentence))
    return sentence


def build_vocabulary(file_name):
    """
        This a funcking bad implementation, but I cannot find a better way. Fuck
    """
    corpus = read_data(file_name)
    vocabulary = dict()
    for s in corpus:
        for w in s:
            if w != TOKEN:
                if w in vocabulary:
                    vocabulary[w] += 1
                else:
                    vocabulary[w] = 1
    vocabulary = list(vocabulary.items())
    vocabulary = sorted(
        vocabulary, key=lambda x: x[1], reverse=True)  # decrease ord
    words, _ = list(zip(*vocabulary))  # get first element
    vocabulary = dict(zip(words, range(
        1,
        len(words) + 1)))  # convert to id, reserver 0 for padding
    vocabulary[TOKEN] = 0
    return vocabulary


def doc2id(filename, word_to_id, time_step):
    """
        retrun:
            a list of ids
    """
    doc = read_data(file_name=filename)
    doc_id = np.zeros([time_step])
    n = 0
    for s in doc:
        temp = [word_to_id[word] for word in s if word in word_to_id]
        temp = fix_sequence_length(time_step, temp)
        doc_id = np.vstack((doc_id, temp))
        n += 1
    return doc_id[1:].astype(np.int32)


def generate_X_Y(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def generate_ptb(filepath, time_step, batch_size):
    """
        return:
            [batch_size, text_vector]
    """
    shuffle = 10000
    train_path = os.path.join(filepath, "ptb.train.txt")
    validation_path = os.path.join(filepath, "ptb.valid.txt")
    test_path = os.path.join(filepath, "ptb.test.txt")

    word_to_id = build_vocabulary(train_path)
    train_data = doc2id(train_path, word_to_id, time_step + 1)
    validation_data = doc2id(validation_path, word_to_id, time_step + 1)
    test_data = doc2id(test_path, word_to_id, time_step + 1)
    chunks_train = tf.data.Dataset.from_tensor_slices(train_data)
    chunks_val = tf.data.Dataset.from_tensor_slices(validation_data)
    chunks_test = tf.data.Dataset.from_tensor_slices(test_data)

    train_dataset = chunks_train.map(generate_X_Y)
    val_dataset = chunks_val.map(generate_X_Y)
    test_dataset = chunks_test.map(generate_X_Y)

    train_dataset = train_dataset.shuffle(shuffle).batch(
        batch_size, drop_remainder=True)
    val_dataset = val_dataset.shuffle(shuffle).batch(
        batch_size, drop_remainder=True)
    test_dataset = test_dataset.shuffle(shuffle).batch(
        batch_size, drop_remainder=True)
    return train_dataset, val_dataset, test_dataset, word_to_id
