import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()  

import cnn

VAL_STEP = 8
EPOCHES = 10
GLOBAL_STEP = tf.Variable(initial_value = 0, trainable = False)

handle = tf.placeholder(tf.string, shape = [])

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
        else:
            iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
        # iterator = tf.contrib.eager.Iterator(dataset)

        iterator = tf.data.Iterator.from_string_handle(handle, train.output_types, train.output_shapes)

        imgs, labels = iterator.get_next()
        imgs = tf.reshape(imgs, [-1, imgs.shape[1], imgs.shape[2], 1])

        if eager is not True:
            # iterator_train = iterator.make_initializer(train)
            # iterator_val = iterator.make_initializer(validation)
            # iterator_test = iterator.make_initializer(test)
            iterator_train = train.make_initializable_iterator()
            iterator_test = test.make_initializable_iterator()
            iterator_val = validation.make_initializable_iterator()
        return iterator_train, iterator_val, iterator_test, imgs, labels



def evaluation_criteria(logit, label, scope_name):
    """
            evaluate precision of current model
    """
    with tf.name_scope(scope_name):
        pred = tf.nn.softmax(logits=logit)
        correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(label, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        return accuracy


def build_computitional_graph(data_set, training=True):
    """
        Input:
            dataset = (X, labels)
    """
    logits, loss, opt, cnn1_w, cnn2_w, fc1_w = cnn.neural_network(data_set)
    accuracy = evaluation_criteria(logit=logits, label=data_set[1], scope_name="evaluation")
    summarise_op = cnn.summarise(loss, accuracy, scope_name="summarise_op")
    return logits, loss, opt, cnn1_w, cnn2_w, fc1_w, accuracy, summarise_op

def epoch_train(sess, epoch, step, loss, opt, logits, label, accuracy, iterator_train, iterator_val, summarise_op, writer, saver):
    """
        Training at one epoch.
        It needs tensorflow sess and data initializer.
    """
    start_time = time.time()
    val_List = []
    step_List = []
    total_loss = 0
    batch = 0
    old_step = step
    sess.run([iterator_train.initializer, iterator_val.initializer])
    # train_handle = sess.run([iterator_train.string_handle()])
    train_handle, val_handle = sess.run([iterator_train.string_handle(), iterator_val.string_handle()])
    # print("train_handle is {0}\n val_handle is {1}".format(train_handle, val_handle))
    try:
        while True:
            train_loss, train_summarise, _ = sess.run([loss, summarise_op, opt], feed_dict= {handle : train_handle})
            writer.add_summary(train_summarise, global_step=step)
            step += 1
            total_loss += train_loss
            batch += 1


            if(step - old_step == VAL_STEP):
                old_step = step
                _, final_accuracy = evaluation(sess=sess, epoch=epoch, gstep=step, logits=logits, label=label, accuracy=accuracy,
                            summarise_op=summarise_op, iterator=iterator_val, writer=writer, saver = saver, kind = 'Validation')
                val_List.append(final_accuracy)
                step_List.append(step)
    
    except tf.errors.OutOfRangeError:
        pass

    print('Average loss at epoch {0}:{1}'.format(epoch, total_loss/batch))
    print('Training cost: {0} seconds'.format(time.time() - start_time))
    return step, val_List, step_List

def evaluation(sess, epoch, gstep, logits, label, accuracy, summarise_op, iterator, writer, saver, kind):
    """
        Evaluation based on evaluation_criteria
    """
    start_time = time.time()
    final_accuracy = 0
    sess.run(iterator.initializer)
    val_handle = sess.run([iterator.string_handle()])
    # print("val_handle is {0}".format(val_handle))

    # import pdb; pdb.set_trace()

    final_accuracy = []

    try:
        while True:
            accuracy_batch, summarise_batch = sess.run([accuracy, summarise_op], feed_dict = {handle : val_handle[0]})
            final_accuracy.append(accuracy_batch)
            writer.add_summary(summarise_batch, global_step=gstep)
    except tf.errors.OutOfRangeError:
        pass

    final_accuracy = np.mean(final_accuracy)

    if(kind == 'Validation'):
        if(final_accuracy > 0.93):
            print("save checkpoint at gstep {0}".format(gstep))
            # saver.save(sess, "myckpt/cnn_prototype", gstep)

    print("{0} Accuracy at epoch {1} : {2}".format(kind, epoch, final_accuracy))
    print("Evaluation time cost: {0} seconds".format(time.time()-start_time))
    return gstep, final_accuracy

def train_nn(train_data, val_data, test_data):
    # Tips: using writer and saver
    try:
        os.makedirs('myckpt')
        os.makedirs('myckpt/cnn_prototype')
    except OSError:
        pass
    iterator_train, iterator_val, iterator_test, X, Y = preprocess_data(
        train_data, val_data, test_data, scope_name="preprocess_data")
    logits, loss, opt,  cnn1_w, cnn2_w, fc1_w, accuracy, summarise_op = build_computitional_graph(
            data_set=(X, Y))
    writer = tf.summary.FileWriter('graphs/cnn_prototype', tf.get_default_graph())
    val_result = []
    step_result = []
    # writer = tf.contrib.summary
    with tf.Session() as sess:
        # import pdb; pdb.set_trace()
        sess.run([tf.global_variables_initializer()])
        # train_init, val_init = sess.run([iterator, iterator_val])
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('mpckpt/cnn_prototype/checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        step = GLOBAL_STEP.eval()
        for epoch in range(0, EPOCHES):
            step, val_List, step_List = epoch_train(sess, epoch=epoch, step=step, loss=loss, opt=opt, logits=logits, label=Y, 
                        accuracy=accuracy, iterator_train=iterator_train, iterator_val = iterator_val, summarise_op=summarise_op, writer=writer, saver=saver)
            # evaluation(sess=sess, epoch=epoch, gstep=step, logits=logits, label=Y, accuracy=accuracy,
            #            summarise_op=summarise_op, iterator=iterator_train, writer=writer, saver = saver, kind = 'Training')
            # evaluation(sess=sess, epoch=epoch, gstep=step, logits=logits, label=Y, accuracy=accuracy,
            #            summarise_op=summarise_op, iterator=iterator_val, writer=writer, saver = saver, kind = 'Validation')
            val_result.extend(val_List)
            step_result.extend(step_List)

            cnn.percent(epoch + 1, EPOCHES + 1)
            print('\n')
        sess.close()
    writer.close()
    print("Training finished!")

    print("step_result is {0}".format(step_result))
    print("val_result is {0}".format(val_result))

    plt.plot(step_result, val_result,'-', label = 'validation accuracy')
    plt.xlabel('step')
    plt.ylabel('accuracy')

    plt.savefig('./val1.jpg')

    plt.show()

train, validation, test = cnn.read_data(cnn.MNIST_PATH)
train_nn(train, validation, test)