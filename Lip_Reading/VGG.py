import tensorflow as tf
import time
import numpy as np

# --------------------------cnn model--------------------------------
# We need to use cnn model as embedding to provide vector to LSTM model
def cnn_layer(inputs, receptive_size, stride, number_filter, padding_0,
              scope_name):
    """
            basic convolutional layer
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

def VGG_layer(image, training = True, timestep,eager = False):
    """
        Buliding whole neural network.
        For demo, the computitional graph is following:
        X -> c -> p -> c -> p -> c -> c -> c  -> p  -> f;
        Input:
            dataset = (X, labels)
        return:
         logits, loss, opt, cnn1_w, cnn2_w, fc1_w
    """
    fcs = []

    if training:
        for i in range(timestep):
            cnn1, cnn1_w = cnn_layer(
                inputs = image[:, i, :, :, :],
                receptive_size = 3,
                stride = 1,
                number_filter = 96,
                padding_0 = 'SAME',
                scope_name = "cnn1")
            pooling1 = pooling_layer(
                inputs = cnn1,
                receptive_size = 3,
                stride = 2,
                padding_0 = 'VALID',
                scope_name = "pooling1")
    
            cnn2, cnn2_w = cnn_layer(
                inputs = pooling1,
                receptive_size = 3,
                stride = 2,
                number_filter = 256,
                padding_0 = 'SAME',
                scope_name = "cnn2")
            pooling2 = pooling_layer(
                inputs = cnn2,
                receptive_size = 3,
                stride = 2,
                padding_0 = 'VALID',
                scope_name = "pooling2")
    
            cnn3, cnn3_w = cnn_layer(
                inputs = pooling2,
                receptive_size = 3,
                stride = 1,
                number_filter = 512,
                padding_0 = 'SAME',
                scope_name = "cnn3")
            cnn4, cnn4_w = cnn_layer(
                inputs = cnn3,
                receptive_size = 3,
                stride = 1,
                number_filter = 512,
                padding_0 = 'SAME',
                scope_name = "cnn4") 
    
            cnn5, cnn5_w = cnn_layer(
                inputs = cnn4,
                receptive_size = 3,
                stride = 1,
                number_filter = 512,
                padding_0 = 'SAME',
                scope_name = "cnn5")   
            pooling5 = pooling_layer(
                inputs = cnn5,
                receptive_size = 3,
                stride = 2,
                padding_0 = 'VALID',
                scope_name = "pooling5")

            fc6, fc6_w = fc_layer(inputs = pooling5, out_dim = 512, scope_name="fc1");
            fcs.append(fc6)
            
        fcs = tf.transpose(tf.concat(0, fcs), perm = [1, 0, 2])

        return fcs, fc6_w

# ------------------------------LSTM model------------------------------------
def rnn(scope_name, cell1, cell2, cell3, cell_state1, cell_state2, cell_state3, X, time_step, number_units, batch_size):
    """
        return:
            hidden_state, probabilities, logit
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        if cell_state1 is not None:
            cell_state1 = cell_state1[0]
            cells_output = []
            for t in range(0, time_step):
                output1, cell_state1 = cell1(X[:, t, :], cell_state1)
                output2, cell_state2 = cell2(output1, cell_state2)
                output3, cell_state3 = cell3(output2, cell_state3)
                cells_output.append(output3)
        else:
            output1, cell_state1 = cell1(tf.transpose(X, (1, 0, 2)))
            output2, cell_state2 = cell2(output1, cell_state2)
            cells_output, cell_state3 = cell3(output2, cell_state3)

    return cell1, cell2, cell3, cell_state1, cell_state2, cell_state3, cells_output

# The cnn batch_size = LSTM batch_size * time_step!
# The cnn batch_size = LSTM batch_size * time_step!
# The cnn batch_size = LSTM batch_size * time_step!
# Because we need to embed the image to vector which shape is [LSTM batch_size, time_step, vec_size(512)], 
# and the image shape is [cnn batch_size, vec_size(512)];
# So when time_step is 8 the image source need sort as [image-1~8, image-2~9, ... ,image-batch_size~batch_size+7]
def Visual_LSTM_layer(image, cell1, cell2, cell3, state1, state2, state3, time_step, number_units, batch_size):
    """
        Pleas try to understand this computation tricks.
        Use eager model to see how tensor flows.
        return:
            final_output, hidden_state, loss, opt, iterator_train, iterator_val, iterator_test
    """
    embeded_source, _ = VGG_layer(image)
    # embeded_source = tf.resize(embeded_source, shape = [:, time_step, :])

    cell1, cell2, cell3, cell_state1, cell_state2, cell_state3, cells_output = rnn(
                                        "Visual_LSTM", cell1, cell2, cell3, state1, state2, state3, 
                                        embeded_source, time_step, number_units, batch_size)
    return cell1, cell2, cell3, cell_state1, cell_state2, cell_state3, cells_output, embeded_source