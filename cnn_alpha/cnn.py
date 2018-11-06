# encoding=utf-8
"""
    Author: Barid AI
    This is a basic prototype of convolutional neural network.
    Requirments:
        1, mnist_utils, a utility for parsing data is provided in the same folder. Pleas read it first in briefly.
        2, tensorflow 1.10+, python 3+, numpy, matplotlib
    Tips:
        1. In tensorflow, you can use "python pdb" for debug purpose,
            I strongly recommand to learn it, it can only cost you 10 mins.
        2. Tensor in tensorflow is actually a numpy adding dimension infomation compared to numpy,
            and an tensor can be transform to a numpy by using "numpy = numpy.array(tensor)"
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# from sklearn.metrics import confusion_matrix
import time
# from datetime import timedelta
# import math
import mnist_utils  # Try to understand the import hierarchy in python.
import os
# ============================eager model============================
# Briefly, it's the same as general model,
# but you can print out op and tensor in this model that it's not supported in general model.
# So, turn on it for coding, turn off it for final training
# import tensorflow.contrib.eager as tfe  # start eager model.
# tfe.enable_eager_execution()


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


def plot_image(image_set, cls_true, cls_predict=None, show_num=9):
    """
        show images in dataset.
        Using it then you can understand it!
    """
    # if image_set.shape[0] >= show_num or image_set.shape[0] is None:
    #     show_num = 9
    # else:
    #     show_num = image_set.shape[0]
    imgs = np.array(image_set[:show_num])
    cls_true = np.array(cls_true[:show_num])
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            np.reshape(imgs[i], (imgs[i].shape[0], imgs[i].shape[1])),
            cmap='binary')
        if cls_predict is None:
            xlabel = "True: {0}".format(np.where(cls_true[i] == 1)[0][0])
        else:
            pre = np.array(cls_predict[i])
            xlabel = "True: {0}, Pre: {1}".format(
                np.where(cls_true[i] == 1)[0][0],
                np.where(pre == 1)[0][0])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    return plt


# ================main mode===============
# ~~~~~~~~~~__init__~~~~~~~~~~
MNIST_PATH = "/Users/barid/Documents/batch_train_data/image-MNIST"
BATCH_SIZE = 128
TRAIN_PRO = 0.75
LR = 0.001
GLOBAL_STEP = tf.Variable(initial_value=0, trainable=False)
EPOCHES = 5


def read_data(MNIST_PATH):
    """
        data struct (img, label)
    """
    train, val, test = mnist_utils.get_mnist_dataset_local(
        batch_size=BATCH_SIZE, local_path=MNIST_PATH)
    return train, val, test


# ~~~~~~~~~~~~~~show images~~~~~~~~~~~
# train, val, test = read_data(MNIST_PATH)
# imgs, labels = preprocess_data(train, True)
# figures = plot_image(imgs, labels)  # show sample images
# figures.show()

# ~~~~~~~~~~~~~preprocess data~~~~~~~~~~


def preprocess_data(train, validation, test, scope_name, eager=False):
    """
        return:
            data_iterator, data, labels
    """
    # Dataset or tensor is a typical data struct in tensorflow, however we say "xiao huo che" in our meeting.
    # I just recognize I have made some confusion in group meeting, so let me explain here again
    # In tensroflow, we basicly have two methods to genarate tensor.
    # 1. Recall in python, we can use "yield" to generate iterator or generator.
    #       Acutally, tensorflow can catch this iterator, following is an example:
    #           dataset = tf.data.Dataset.from_generator(yield, data_type, data_shape)
    #           dataset = dataset..make_initializable_iterator()
    #           item =  dataset.get_next()
    # 2. We use it here.

    with tf.name_scope(scope_name):
        if eager:
            iterator_train = train.make_one_shot_iterator()
            iterator_test = test.make_one_shot_iterator()
            iterator_val = validation.make_one_shot_iterator()
            imgs, labels = iterator_train.get_next()
        else:
            iterator = tf.data.Iterator.from_structure(train.output_types,
                                                       train.output_shapes)
            iterator_train = iterator.make_initializer(train)
            iterator_val = iterator.make_initializer(validation)
            iterator_test = iterator.make_initializer(test)
            # iterator = tf.contrib.eager.Iterator(dataset)
            imgs, labels = iterator.get_next()
        imgs = tf.reshape(imgs, [-1, imgs.shape[1], imgs.shape[2], 1])

        return iterator_train, iterator_val, iterator_test, imgs, labels


# ~~~~~~~~~~~~layer construction~~~~~~~~~
# Tips:
#   1, In terms of initializer, "w" uses tf.truncated_normal_initializer, "b" uses tf.random_normal_initializer
#   2. In terms of "scope", it should try "tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE)" at first,
#       instead of "tf.name_scope(scope_name)", however if you can make sure there is no variable for "reusing",
#       you can use "tf.name_scope(scope_name)""


def cnn_layer(inputs, receptive_size, stride, number_filter, padding_0,
              scope_name):
    """
            basic convolutional layer
            OP! OP! OP!
            Return:
                relu_cnn op, kerne
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        channel = inputs.shape[-1]
        kernel = tf.get_variable(
            scope_name + "_kernel",
            shape=[receptive_size, receptive_size, channel, number_filter],
            initializer=tf.truncated_normal_initializer())
        bias = tf.get_variable(
            scope_name + "_bias",
            shape=[number_filter],
            initializer=tf.random_normal_initializer())
        cnn = tf.nn.conv2d(
            input=inputs,
            filter=kernel,
            strides=[1, stride, stride, 1],
            padding=padding_0)
        relu_cnn = tf.nn.relu(cnn + bias)
        return relu_cnn, kernel


def pooling_layer(inputs, receptive_size, stride, padding_0, scope_name):
    """
        basic pooling_layer
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        pooling = tf.nn.max_pool(
            inputs,
            ksize=[1, receptive_size, receptive_size, 1],
            strides=[1, stride, stride, 1],
            padding=padding_0)

        return pooling


def fc_layer(inputs, out_dim, scope_name):
    """
        basic fc layer
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        in_dim = inputs.shape[1] * inputs.shape[2] * inputs.shape[3]
        inputs = tf.reshape(inputs, [-1, in_dim])
        w = tf.get_variable(
            scope_name + "_w",
            shape=[in_dim, out_dim],
            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(
            scope_name + "_bias",
            shape=[out_dim],
            initializer=tf.constant_initializer())
        fc = tf.matmul(inputs, w) + b
        return fc, w


def softmax_with_dropout(inputs, classes, drop_out, scope_name):
    """
        For trainning, set drop_out value in [0,1], probabily 0.4.
        For test and prediction, set drop_out value to 1.
        Return:
                logits
    """
    #  In tensorflow, typically, we don't use "tf.nn.softmax" in softmax layer, it only needs to compute logits.
    #  y = softmax(logits), for more details see softmax and logistic regression inference.
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        inputs = tf.nn.dropout(tf.nn.relu(inputs), keep_prob=TRAIN_PRO)
        in_dim = inputs.shape[-1]
        w = tf.get_variable(
            scope_name + "_w",
            shape=[in_dim, classes],
            initializer=tf.truncated_normal_initializer())
        b = tf.get_variable(
            scope_name + "_b",
            shape=[classes],
            initializer=tf.constant_initializer())
        logits = tf.matmul(inputs, w) + b
        return logits


# ~~~~~~~~~~loss function~~~~


def loss_function(logit, label, scope_name):
    """
        loss function
    """
    with tf.variable_scope(scope_name):
        # go back to softmax, and try to understand this.
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logit, labels=label)
        loss = tf.reduce_mean(entropy)
        return loss


# ~~~~~~~~~~optimizer for loss~~~~~~~~~~~~


def optimizer(loss, gstep, lr, scope_name):
    """
        adm optimizer
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        opt = tf.train.AdamOptimizer(lr).minimize(loss=loss, global_step=gstep)
        return opt


def summarise(loss, accuracy, scope_name):
    """
        plot loss and accuracy for analyseing
    """
    with tf.name_scope(scope_name):
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.histogram("loss", loss)
    return tf.summary.merge_all()


# ~~~~~~~~~~~~~hierarchy for main model ~~~~~~~


def neural_network(data_set, training=True, eager=False):
    """
        Buliding whole neural network.
        For demo, the computitional graph is following:
        X -> c -> p -> c -> p -> f -> softmax -> Yre
        Input:
            dataset = (X, labels)
        return:
         logits, loss, opt, cnn1_w, cnn2_w, fc1_w
    """
    if training:
        classes = data_set[1].shape[-1]
        cnn1, cnn1_w = cnn_layer(
            data_set[0],
            receptive_size=5,
            stride=1,
            number_filter=32,
            padding_0='SAME',
            scope_name="cnn1")
        pooling1 = pooling_layer(
            inputs=cnn1,
            receptive_size=2,
            stride=2,
            padding_0='VALID',
            scope_name="pooling1")

        cnn2, cnn2_w = cnn_layer(
            pooling1,
            receptive_size=5,
            stride=1,
            number_filter=64,
            padding_0='SAME',
            scope_name="cnn2")
        pooling2 = pooling_layer(
            inputs=cnn2,
            receptive_size=2,
            stride=2,
            padding_0='VALID',
            scope_name="pooling2")

        fc1, fc1_w = fc_layer(inputs=pooling2, out_dim=1024, scope_name="fc1")
        logits = softmax_with_dropout(
            fc1, classes=classes, drop_out=TRAIN_PRO, scope_name="logits")
        loss = loss_function(
            logit=logits, label=data_set[1], scope_name="loss")
        opt = optimizer(
            loss=loss, gstep=GLOBAL_STEP, lr=LR, scope_name='optimizer')
        return logits, loss, opt, cnn1_w, cnn2_w, fc1_w


def evaluation_criteria(logit, label, scope_name):
    """
            evaluate precision of current model
    """
    with tf.name_scope(scope_name):
        pred = tf.nn.softmax(logits=logit)
        correct_pred = tf.equal(
            tf.argmax(pred, axis=1), tf.argmax(label, axis=1))
        accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))
        return accuracy


# ~~~~~~~~~~~~~~build computitional graph~~~~~~~~


def build_computitional_graph(data_set, training=True):
    """
        Input:
            dataset = (X, labels)
    """
    logits, loss, opt, cnn1_w, cnn2_w, fc1_w = neural_network(data_set)
    accuracy = evaluation_criteria(
        logit=logits, label=data_set[1], scope_name="evaluation")
    summarise_op = summarise(loss, accuracy, scope_name="summarise_op")
    return logits, loss, opt, cnn1_w, cnn2_w, fc1_w, accuracy, summarise_op


def epoch_train(sess, epoch, step, loss, opt, iterator, summarise_op, writer,
                saver):
    """
        Training at one epoch.
        It needs tensorflow sess and data initializer.
    """
    start_time = time.time()
    total_loss = 0
    batch = 0
    sess.run([iterator])
    try:
        while True:
            train_loss, train_summarise, _ = sess.run(
                [loss, summarise_op, opt])
            writer.add_summary(train_summarise, global_step=step)
            step += 1
            total_loss += train_loss
            batch += 1
    except tf.errors.OutOfRangeError:
        pass
    saver.save(sess, "checkpoints/cnn_prototype", step)
    print('Average loss at epoch {0}:{1}'.format(epoch, total_loss / batch))
    print('Training cost: {0} seconds'.format(time.time() - start_time))
    return step


def evaluation(sess, epoch, gstep, logits, label, accuracy, summarise_op,
               iterator, writer):
    """
        Evaluation based on evaluation_criteria
    """
    start_time = time.time()
    final_accuracy = 0
    sess.run([iterator])
    # import pdb; pdb.set_trace()
    try:
        while True:
            accuracy_batch, summarise_batch = sess.run(
                [accuracy, summarise_op])
            final_accuracy += accuracy_batch
            final_accuracy = final_accuracy / 2
            writer.add_summary(summarise_batch, global_step=gstep)
    except tf.errors.OutOfRangeError:
        pass

    print("Accuracy at epoch {0} : {1}".format(epoch, final_accuracy))
    print("Evaluation time cost: {0} seconds".format(time.time() - start_time))
    return gstep


def train_nn(train_data, val_data, test_data):
    # Tips: using writer and saver
    try:
        os.makedirs('checkpoints')
        os.makedirs('checkpoints/cnn_prototype')
    except OSError:
        pass
    iterator_train, iterator_val, iterator_test, X, Y = preprocess_data(
        train_data, val_data, test_data, scope_name="preprocess_data")
    logits, loss, opt, cnn1_w, cnn2_w, fc1_w, accuracy, summarise_op = build_computitional_graph(
        data_set=(X, Y))
    writer = tf.summary.FileWriter('graphs/cnn_prototype',
                                   tf.get_default_graph())
    # writer = tf.contrib.summary
    with tf.Session() as sess:
        # import pdb; pdb.set_trace()
        sess.run([tf.global_variables_initializer()])
        # train_init, val_init = sess.run([iterator, iterator_val])
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(
            'checkpoints/cnn_prototype/checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        step = GLOBAL_STEP.eval()
        for epoch in range(0, EPOCHES):
            step = epoch_train(
                sess,
                epoch=epoch,
                step=step,
                loss=loss,
                opt=opt,
                iterator=iterator_train,
                summarise_op=summarise_op,
                writer=writer,
                saver=saver)
            evaluation(
                sess=sess,
                epoch=epoch,
                gstep=step,
                logits=logits,
                label=Y,
                accuracy=accuracy,
                summarise_op=summarise_op,
                iterator=iterator_val,
                writer=writer)
            percent(epoch + 1, EPOCHES + 1)
            print('\n')
        sess.close()
    writer.close()
    print("Training finished!")


train, validation, test = read_data(MNIST_PATH)
train_nn(train, validation, test)

#  #########################################eager model############
# show images, but only support in eager model
# print(
#     "This is the eager model for debug purpose, if you are going to train the model pls comment eager model and\
#         tfe.enable_eager_execution() which is at the top of this file, and then uncomment train_nn()"
# )
# iterator_train, iterator_val, iterator_test, imgs, labels = preprocess_data(
#     train, validation, test, scope_name="preprocess_data", eager=True)
# figure = plot_image(imgs, labels)
# data_set = (imgs, labels)
# classes = data_set[1].shape[-1]
# cnn1, cnn1_w = cnn_layer(
#     data_set[0],
#     receptive_size=5,
#     stride=1,
#     number_filter=32,
#     padding_0='SAME',
#     scope_name="cnn1")
# pooling1 = pooling_layer(
#     inputs=cnn1,
#     receptive_size=2,
#     stride=2,
#     padding_0='VALID',
#     scope_name="pooling1")
#
# cnn2, cnn2_w = cnn_layer(
#     pooling1,
#     receptive_size=5,
#     stride=1,
#     number_filter=64,
#     padding_0='SAME',
#     scope_name="cnn2")
# pooling2 = pooling_layer(
#     inputs=cnn2,
#     receptive_size=2,
#     stride=2,
#     padding_0='VALID',
#     scope_name="pooling2")
#
# fc1, fc1_w = fc_layer(inputs=pooling2, out_dim=1024, scope_name="fc1")
# logits = softmax_with_dropout(
#     fc1, classes=classes, drop_out=TRAIN_PRO, scope_name="logits")
# loss = loss_function(logit=logits, label=data_set[1], scope_name="loss")
# eval = evaluation_criteria(logit=logits, label=labels, scope_name="eval")
# with tf.GradientTape() as tape:
#     error = loss
#     grad = tape.gradient(target=error, sources=[logits])
#     # print(error, grad)
# checkpoint = tf.train.Checkpoint(cnn1_w=cnn1_w)
# status = checkpoint.restore(
#     tf.train.latest_checkpoint("checkpoints/cnn_prototype/"))
