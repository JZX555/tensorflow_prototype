# encoding=utf8

import tensorflow as tf
import numpy as np
import os


class DataParser():
    """
        if model= X
        file_name is a dictionary: {"train": file_name, "val":file_name, "test":file_name}
    """

    def __init__(self,
                 data_path,
                 file_name,
                 time_step,
                 batch_size,
                 model="X",
                 local=True,
                 token='\n',
                 marke='<eos>'):
        self.data_path = data_path
        self.local = local
        self.time_step = time_step
        self.batch_size = batch_size
        self.token = token
        self.marker = token
        self.file_name = file_name
        self.model = model

    def read_data(self, data_path):
        if self.local:
            with open(data_path, 'r', encoding='utf8') as f:
                for k, v in enumerate(f):
                    yield v.replace(self.token, self.marker).split()
        else:
            with tf.gfile.GFile(name=self.file_name) as f:
                return f.read().replace(self.token, self.marker).split()
        pass

    def fix_sequence_length(self, sentence):
        length = len(sentence)
        if length >= self.time_step:
            sentence = sentence[:self.time_step]
        else:
            for i in range(length, self.time_step):
                sentence = np.concatenate((np.zeros((1)), sentence))
        return sentence

    def build_vocabulary(self, train_path):
        """
            This a funcking bad implementation, but I cannot find a better way. Fuck
        """
        corpus = self.read_data(train_path)
        vocabulary = dict()
        for s in corpus:
            for w in s:
                if w != self.token:
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
        vocabulary[self.token] = 0
        return vocabulary

    def doc2id(self, data_path):
        """
            retrun:
                a list of ids
        """
        doc = self.read_data(data_path)
        doc_id = np.zeros([self.time_step])
        n = 0
        for s in doc:
            temp = [
                self.word_to_id[word] for word in s if word in self.word_to_id
            ]
            temp = self.fix_sequence_length(temp)
            doc_id = np.vstack((doc_id, temp))
            n += 1
        return doc_id[1:].astype(np.int32)

    def generate_X_Y(self, chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    def generate_X(self, chunk):
        return chunk

    def generate_data(self):
        """
            return:
                [batch_size, text_vector]
        """
        shuffle = 10000
        train_dataset = []
        val_dataset = []
        word_to_id = []
        test_dataset = []
        try:
            train_path = os.path.join(self.data_path, self.file_name["train"])
        except Exception:
            print("Please support train data")
        self.word_to_id = self.build_vocabulary(train_path)
        train_data = self.doc2id(train_path)
        if "val" in self.file_name:
            validation_path = os.path.join(self.data_path,
                                           self.file_name['val'])
            validation_data = self.doc2id(validation_path)
        else:
            validation_data = train_data[int(len(train_data) *
                                             0.7):int(len(train_data) * 0.9)]
            train_data = np.delete(
                train_data,
                list(range(int(len(train_data) * 0.7),
                     len(train_data) + 1)), 0)
        if "test" in self.file_name:
            test_path = os.path.join(self.data_path, self.file_name['test'])
            test_data = self.doc2id(test_path)
        else:
            test_data = train_data[int(len(train_data) * 0.9):-1]
            train_data = np.delete(
                train_data,
                list(range(int(len(train_data) * 0.8),
                           len(train_data) + 1)), 0)
        chunks_train = tf.data.Dataset.from_tensor_slices(train_data)
        chunks_val = tf.data.Dataset.from_tensor_slices(validation_data)
        chunks_test = tf.data.Dataset.from_tensor_slices(test_data)
        if self.model == "X_Y":
            train_dataset = chunks_train.map(self.generate_X_Y)
            val_dataset = chunks_val.map(self.generate_X_Y)
            test_dataset = chunks_test.map(self.generate_X_Y)
        else:
            train_dataset = chunks_train.map(self.generate_X)
            val_dataset = chunks_val.map(self.generate_X)
            test_dataset = chunks_test.map(self.generate_X)
        train_dataset = train_dataset.shuffle(shuffle).batch(
            self.batch_size, drop_remainder=True)

        val_dataset = val_dataset.shuffle(shuffle).batch(
            self.batch_size, drop_remainder=True)

        test_dataset = test_dataset.shuffle(shuffle).batch(
            self.batch_size, drop_remainder=True)
        return train_dataset, val_dataset, test_dataset, word_to_id


if __name__ == '__main__':
    DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus_cn2en/'
    dp = DataParser(DATA_PATH, {'train': 'data.cn-en.s'}, 16, 64)
    a = dp.generate_data()
