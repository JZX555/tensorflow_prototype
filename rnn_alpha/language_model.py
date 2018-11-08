# encoding=utf8
"""

    Author: Barid Ai
    This is a basic rnn prototypeself.
    In this model, it presents how to use rnn to generate/predict words.

    Gernerally, in nlp tasks, we have an standar pipline, and fc layer isn't a standar procedure
    compared to CNN, as you can see below:
    1. preprocess data
    2. word embedding
    3. neural network(alway using rnn architecture)
    4. loss function
    5. optimizer


    Tips:
        Terminoloies in literature and tensorflow differ from each other, and it causes some confusion, so I will
        explain here. Firstly, when we discuss cells like 1000 cells in literature, it refers to the dimension of outputs from hidden state in most of time,
        which is represented as softmax outputs in tensorflow. Secondly, number of units is an import item in tensorflow compared to literature,
        because in literature number of units is rarely mentioned but it is a signiture of rnn related function. And then time step or number of cells in tensorflow refers the length of
        sequence in literature.

        LSTMStateTuple

"""

import sys
sys.path.insert(0, '/home/vivalavida/workspace/batch_data/corpus-ptb')
# sys.path.insert(0, '/Users/barid/Documents/workspace/batch_data/corpus-ptb')
import tensorflow as tf
import ptb_reader
import tensorflow.contrib as tf_contrib
import time
import os

# ###########eager model#####
# tf_contrib.eager.enable_eager_execution()
# #############################
# DATA_PATH = '/Users/barid/Documents/workspace/batch_data/corpus-ptb/data/'
DATA_PATH = sys.path[0] + '/data'
BATCH_SIZE = 64
LEARNING_RATE = 0.01
KEEP_PROB = 0.4
GLOBAL_STEP = tf.Variable(initial_value=0, trainable=False)
NUMBER_EPOCH = 20
CLIPPING = 5
# ~~~~~~~~~~~~~~ basic LSTM configure~~~~~~~~~~~~~~~~~
TIME_STEP = 8  # number of hidden state, it also means the lenght of input and output sentence
NUMBER_UNITS = 1024  # dimension of each hidden state
# ~~~~~~~~~~~~~ embedding configure ~~~~~~~~~~~~~~~~~
EMBEDDING_SIZE = NUMBER_UNITS


# =========================visualization=========================
def percent(current, total):
    """
        just for fun
    """
    import sys
    if current == total - 1:
        current = total
    bar_length = 20
    hashes = '#' * int(current / total * bar_length)
    spaces = ' ' * (bar_length - len(hashes))
    sys.stdout.write(
        "\rPercent: [%s] %d%%" % (hashes + spaces, int(100 * current / total)))


# ========================model======================================
def get_pdb_data(batch_size, data_path):
    train, val, test, vocabulary = ptb_reader.generate_ptb(
        DATA_PATH, TIME_STEP, BATCH_SIZE)
    return train, val, test, vocabulary


def preprocess_data(scope_name, train, val, test, time_step, eager=False):
    """
        return:
            iterator_train, iterator_val, iterator_test, source_sentence, target_sentence
    """
    with tf.name_scope(scope_name):
        if eager:
            iterator_train = train.make_one_shot_iterator()
            iterator_val = val.make_one_shot_iterator()
            iterator_test = test.make_one_shot_iterator()
            source_sentence, target_sentence = iterator_train.get_next()
        else:
            iterator = tf.data.Iterator.from_structure(
                output_types=train.output_types,
                output_shapes=train.output_shapes)
            iterator_train = iterator.make_initializer(train)
            iterator_val = iterator.make_initializer(val)
            iterator_test = iterator.make_initializer(test)
            source_sentence, target_sentence = iterator.get_next()
        # source_sentence = tf.reshape(source_sentence, [-1, source_sentence.shape[1], source_sentence.shape[2]])
        # target_sentence = tf.reshape(target_sentence, [-1, target_sentence.shape[1], target_sentence.shape[2]])
        # if eager is not True:
        #     iterator_train = iterator.make_initializer(train)
        #     iterator_val = iterator.make_initializer(val)
        #     iterator_test = iterator.make_initializer(test)
    return iterator_train, iterator_val, iterator_test, source_sentence, target_sentence


def embedding(scope_name, X, embedding_size, vocabulary):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        vocabulary_size = len(vocabulary)
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=[vocabulary_size, embedding_size],
            initializer=tf.random_normal_initializer())
        embed = tf.nn.embedding_lookup(embedding_matrix, X, name="embed")

        return embed


def rnn(scope_name, cell, cell_state, X, time_step, number_units, batch_size):
    """
        return:
            hidden_state, probabilities, logit
    """
    # import pdb; pdb.set_trace()
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        cell_output = []
        for t in range(0, time_step):
            output, cell_state = cell(X[:, t, :], cell_state)
            cell_output.append(output)
    return cell, cell_state, cell_output


def softmax_with_dropout(scope_name,
                         state_output,
                         time_step,
                         number_units,
                         vocabulary_size,
                         keep_prob=1):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        state_output = tf.nn.dropout(state_output, keep_prob)
        logits_w = tf.get_variable(
            "logits_w_" + str(time_step),
            shape=[number_units, vocabulary_size],
            initializer=tf.truncated_normal_initializer())
        logits_b = tf.get_variable(
            "logits_b_" + str(time_step),
            shape=[vocabulary_size],
            initializer=tf.random_normal_initializer())
        logit = tf.nn.xw_plus_b(state_output, logits_w, logits_b)
        softmax_output = tf.nn.softmax(logit)
    return softmax_output


def loss_fucntion(scope_name, softmax, Y, batch_size, timestep):
    with tf.name_scope(scope_name):
        loss = tf_contrib.seq2seq.sequence_loss(
            softmax,
            Y,
            tf.ones([batch_size, timestep], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True)
    return loss


def optimizer(scope_name, loss, lr, clipping):
    """
        In tensorflow, considering the computational efficiency,
        it recommends to use a Truncated Backpropagation as refered below:
        "By design, the output of a recurrent neural network (RNN) depends on arbitrarily distant inputs. Unfortunately,
        this makes backpropagation computation difficult. In order to make the learning process tractable,
        it is common practice to create an "unrolled" version of the network,
        which contains a fixed number (num_steps) of LSTM inputs and outputs.
        The model is then trained on this finite approximation of the RNN.
        This can be implemented by feeding inputs of length num_steps
        at a time and performing a backward pass after each such input block."


        We use batch normolization here, and gradient cliping method to avoid gradient explosion.


    """
    with tf.variable_scope(scope_name):
        cost = tf.reduce_sum(loss)
        train_var = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(cost, train_var), 5)
        optimizer = tf.train.GradientDescentOptimizer(lr)
        opt = optimizer.apply_gradients(zip(gradients, train_var))
        # opt = optimizer.minimize(cost)
    return opt


def network(train, validation, test, vocabulary, cell, state, time_step,
            number_units, embedding_size, batch_size, keep_prob):
    """
        Pleas try to understand this computation tricks.
        Use eager model to see how tensor flows.
        return:
            final_output, hidden_state, loss, opt, iterator_train, iterator_val, iterator_test
    """
    vocabulary_size = len(vocabulary)
    iterator_train, iterator_val, iterator_test, source_sentence, target_sentence = preprocess_data(
        "preprocess_data", train, validation, test, time_step)
    embeded_source = embedding("embedding", source_sentence, embedding_size,
                               vocabulary)
    cell, cell_state, cell_output = rnn("rnn", cell, state, embeded_source,
                                        time_step, number_units, batch_size)
    cell_output = tf.reshape(tf.concat(cell_output, 1), [-1, number_units])
    softmax_output = softmax_with_dropout("softmax", cell_output, time_step,
                                          number_units, vocabulary_size,
                                          keep_prob)

    softmax_output = tf.reshape(softmax_output,
                                [-1, time_step, vocabulary_size])
    return cell, cell_state, softmax_output, iterator_train, iterator_val, iterator_test, source_sentence, target_sentence


def summarize(scope_name, loss, accuracy):
    """
        return:
            summarize_op
    """
    with tf.name_scope(scope_name):
        tf.summary.scalar("loss", tf.reduce_mean(loss))
        tf.summary.scalar("accuracy", accuracy)
        return tf.summary.merge_all()


def evaluation_criteria(scope_name, predict, label):

    with tf.name_scope(scope_name):
        correct_pre = tf.equal(
            tf.argmax(predict, axis=2), tf.cast(label, tf.int64))
        accuracy = tf.reduce_sum(tf.cast(correct_pre, tf.float32))
        return accuracy


def build_computational_grap(train, validation, test, vocabulary, cell, state,
                             time_step, number_units, embedding_size,
                             batch_size, keep_prob, lr, clipping):
    cell, cell_state, softmax_output, iterator_train, iterator_val, iterator_test, source_sentence, target_sentence = network(
        train, validation, test, vocabulary, cell, state, time_step,
        number_units, embedding_size, batch_size, keep_prob)
    loss = loss_fucntion("loss", softmax_output, target_sentence, batch_size,
                         time_step)
    opt = optimizer("optimizer", loss, lr, clipping)
    accuracy = evaluation_criteria("epoch_evaluation", softmax_output,
                                   target_sentence)
    summarize_op = summarize('summarize', loss, accuracy)
    return cell, cell_state, softmax_output, iterator_train, iterator_val, iterator_test, source_sentence, target_sentence, loss, opt, summarize_op, accuracy


def epoch_train(sess, epoch, step, init, state, time_step, loss, opt,
                summarize_op, writer, saver):
    """
        Training at one epoch.
        It needs tensorflow sess and data initializer.
    """
    start_time = time.time()
    total_loss = 0
    batch = 0
    sess.run([init])
    try:
        while True:
            epoch_loss, epoch_summarize, _ = sess.run(
                [loss, summarize_op, opt])
            writer.add_summary(epoch_summarize, global_step=step)
            step += 1
            total_loss += epoch_loss
            batch += 1
    except tf.errors.OutOfRangeError:
        pass
    saver.save(sess, "checkpoints/rnn_alpha", step)
    print('Average loss at epoch {0}:{1}'.format(epoch, total_loss / batch))
    print('Training cost: {0} seconds'.format(time.time() - start_time))
    return step


def epoch_evaluation(sess, epoch, step, init, predict, label, accuracy,
                     summarize_op, writer):
    """
        Evaluation based on evaluation_criteria
    """
    start_time = time.time()
    final_accuracy = 0
    sess.run([init])
    try:
        epoch_accuracy, epoch_summarize = sess.run([accuracy, summarize_op])
        final_accuracy += epoch_accuracy
        final_accuracy = final_accuracy / 2
        writer.add_summary(epoch_summarize, global_step=step)
    except tf.errors.OutOfRangeError:
        pass

    print("Accuracy at epoch {0} : {1}".format(epoch, final_accuracy))
    print("Evaluation time cost: {0} seconds".format(time.time() - start_time))
    return step


def train_nn(train, validation, test, vocabulary, time_step, number_units,
             embedding_size, batch_size, keep_prob, number_epoch, gstep, lr,
             clipping):
    try:
        os.makedirs("checkpoints")
        os.makedirs("checkpoints/rnn_alpha")
    except OSError:
        pass
    writer = tf.summary.FileWriter('graphs/rnn_alpha', tf.get_default_graph())
    cell = tf.nn.rnn_cell.LSTMCell(number_units)
    state = cell.zero_state(batch_size, dtype=tf.float32)
    cell, cell_state, cell_output, iterator_train, iterator_val, iterator_test, source_sentence, target_sentence, loss, opt, summarize_op, accuracy = build_computational_grap(
        train, validation, test, vocabulary, cell, state, time_step,
        number_units, embedding_size, batch_size, keep_prob, lr, clipping)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        ckpts = tf.train.get_checkpoint_state('checkpoints/rnn_alpha')
        if ckpts and ckpts.model_checkpoint_path:
            saver.restore(sess, ckpts.model_checkpoint_path)
        step = gstep.eval()
        for e in range(0, number_epoch):
            step = epoch_train(sess, e, step, iterator_train, cell_state,
                               time_step, loss, opt, summarize_op, writer,
                               saver)
            epoch_evaluation(sess, e, step, iterator_val, cell_output,
                             target_sentence, accuracy, summarize_op, writer)
            percent(e + 1, number_epoch)
            print('\n')

        sess.close()
    writer.close()
    print("Training finished!")


train, validation, test, vocabulary = get_pdb_data(BATCH_SIZE, DATA_PATH)
train_nn(train, validation, test, vocabulary, TIME_STEP, NUMBER_UNITS,
         EMBEDDING_SIZE, BATCH_SIZE, KEEP_PROB, NUMBER_EPOCH, GLOBAL_STEP,
         LEARNING_RATE, CLIPPING)
#  #########################################eager model############
# print(
#     "This is the eager model for debug purpose, if you are going to train the model pls comment eager model and\
#         tfe.enable_eager_execution() which is at the top of this file, and then uncomment train_nn()"
# )
# iterator_train, iterator_val, iterator_test, source_sentence, target_sentence = preprocess_data(
#     "pd", train, validation, test, TIME_STEP, True)
# embed = embedding("embedding", source_sentence, EMBEDDING_SIZE, vocabulary)
# vocabulary_size = len(vocabulary)
# cell = tf.nn.rnn_cell.LSTMCell(NUMBER_UNITS)
# state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
# cell, cell_state, cell_output = rnn("rnn", cell, state, embed, TIME_STEP,
#                                     NUMBER_UNITS, BATCH_SIZE)
# cell_output = tf.reshape(tf.concat(cell_output, 1), [-1, NUMBER_UNITS])
# softmax_output = softmax_with_dropout("softmax", cell_output, TIME_STEP,
#                                       NUMBER_UNITS, vocabulary_size)
# final_output = tf.reshape(softmax_output, [-1, TIME_STEP, vocabulary_size])
# loss = loss_fucntion("loss", final_output, target_sentence, BATCH_SIZE,
#                      TIME_STEP)
# opt = optimizer("opt", loss)
# eval = evaluation_criteria("eval", final_output, target_sentence)
