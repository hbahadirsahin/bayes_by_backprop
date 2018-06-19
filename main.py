import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from cifar_input import load_dataset


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def weight_variable_for_dropout(shape):
    epsilon = tf.truncated_normal(shape, mean=0,  stddev=0.1, dtype=tf.float32)
    w_mu = tf.truncated_normal(shape, stddev=0.1)
    w_sigma = tf.truncated_normal(shape, stddev=0.1)
    initial = w_mu + tf.log1p(tf.exp(w_sigma)) * epsilon
    return tf.Variable(initial), tf.Variable(w_mu), tf.Variable(w_sigma)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def bias_variable_for_dropout(shape):
    epsilon = tf.random_uniform(shape, dtype=tf.float32)
    b_mu = tf.constant(0.1, shape=shape)
    b_sigma = tf.constant(0.1, shape=shape)
    initial = b_mu + tf.exp(b_sigma) * epsilon
    return tf.Variable(initial), tf.Variable(b_mu)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=stride, padding='SAME')


def max_pool(x, filter_size, stride):
    return tf.nn.max_pool(x, filter_size, strides=stride, padding='SAME')


constant = -0.5 * np.log(2 * np.pi)


def log_gaussian(x, mu, sigma):
    return constant - np.log(np.abs(sigma)) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    std = tf.log1p(tf.exp(logsigma))
    return constant - tf.log(tf.abs(std)) - (x - mu) ** 2 / (2 * std ** 2)


def run_variational_network():
    xtrain, ytrain, xtest, ytest = load_dataset()

    train_size = xtrain.shape[0]
    max_epoch = 60
    num_samples = 1
    batch_size = 250
    num_labels = 10
    pi = 1/4
    rho_1 = np.exp(-1)
    rho_2 = np.exp(-7)
    log_pw, log_qw = 0., 0.
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, num_labels])
    keep_prob = tf.constant(0.5, tf.float32)
    is_training = tf.placeholder(tf.bool)


    for _ in range(num_samples):
        # First Convolutional layer
        W_conv1 = weight_variable([3, 3, 3, 32])
        b_conv1 = bias_variable([32])

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)
        h_pool1 = max_pool(h_conv1, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Second Convolutional layer
        W_conv2 = weight_variable([3, 3, 32, 32])
        b_conv2 = bias_variable([32])

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)
        h_pool2 = max_pool(h_conv2, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

        # Flatten before fully connected layer
        h_pool2_flat = tf.contrib.layers.flatten(h_pool2)

        # Weight&Bias initialization for Fully Connected Layer
        W_fc1, W_mu_fc1, w_sigma_fc1 = weight_variable_for_dropout([128, 512])
        b_fc1  = bias_variable([512])
        W_fc2, W_mu_fc2, w_sigma_fc2 = weight_variable_for_dropout([512, 512])
        b_fc2= bias_variable([512])

        tf.add_to_collection('W_mu', W_mu_fc1)
        tf.add_to_collection('W_mu', W_mu_fc2)

        # First Fully Connected Layer
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Second Fully Connected Layer
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

        # Output layer
        W_fc3 = weight_variable([512, num_labels])
        b_fc3 = bias_variable([num_labels])

        output = tf.matmul(h_fc2, W_fc3) + b_fc3

        sample_log_pw, sample_log_qw = 0., 0.

        for W, W_mu, W_sigma in [(W_fc1, W_mu_fc1,w_sigma_fc1),
                                 (W_fc2, W_mu_fc2, w_sigma_fc2)]:

            # Weight prior
            sample_log_pw += tf.reduce_sum(pi * log_gaussian(W, 0., rho_1) + (1-pi) * log_gaussian(W, 0., rho_2))

            # Approximation
            sample_log_qw += tf.reduce_sum(log_gaussian_logsigma(W, W_mu, W_sigma))

        log_pw += sample_log_pw
        log_qw += sample_log_qw

    log_qw /= num_samples
    log_pw /= num_samples

    print("Loss")
    with tf.name_scope("LOSS"):
        dropout_loss = tf.reduce_sum(1./train_size/float(batch_size) * (log_qw - log_pw)) / float(batch_size)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y)) + dropout_loss

    print("Accuracy")
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    print("Optimizer")
    with tf.name_scope('OPTIMIZER'):
        batch = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(1e-3, batch*batch_size, train_size, 0.95, staircase=True)
        grad = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

        num_batches = int(train_size / batch_size)
        for epoch in range(max_epoch):
            for batch in range(num_batches):
                batch_offset = (batch * batch_size) % train_size
                batch_data = xtrain[batch_offset:(batch_offset+batch_size)]
                batch_labels = ytrain[batch_offset:(batch_offset+batch_size)]

                train_loss, _, train_accuracy = sess.run([loss, grad, accuracy], feed_dict={x:batch_data, y:batch_labels, is_training:True})

            print("Epoch:", (epoch+1), "/", max_epoch, "- Training accuracy:", train_accuracy, "- Train loss:", train_loss)


        _, test_accuracy = sess.run([grad, accuracy], feed_dict={x: xtest, y: ytest, keep_prob: 1.0})
        print("test accuracy for the stored model:", test_accuracy)
        saver.save(sess, '/model_var_drop')


def run_basic_network():
    xtrain, ytrain, xtest, ytest = load_dataset()

    print(xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)

    train_size = xtrain.shape[0]
    max_epoch = 60
    batch_size = 250
    num_labels = 10

    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, num_labels])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)

    # First Convolutional layer
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x, W_conv1, stride=[1, 2, 2, 1]) + b_conv1)
    h_pool1 = max_pool(h_conv1, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    # Second Convolutional layer
    W_conv2 = weight_variable([3, 3, 32, 32])
    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, stride=[1, 2, 2, 1]) + b_conv2)
    h_pool2 = max_pool(h_conv2, filter_size=[1, 2, 2, 1], stride=[1, 2, 2, 1])

    # Flatten before fully connected layer
    h_pool2_flat = tf.contrib.layers.flatten(h_pool2)

    # Weight/Bias initialization for Fully Connected Layer
    W_fc1 = weight_variable([128, 512])
    b_fc1 = bias_variable([512])
    W_fc2 = weight_variable([512, 512])
    b_fc2 = bias_variable([512])
    tf.add_to_collection('W', W_fc1)
    tf.add_to_collection('W', W_fc2)

    # First Fully Connected Layer
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Second Fully Connected Layer

    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


    # Dropout
    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    # Outut layer
    W_fc3 = weight_variable([512, num_labels])
    b_fc3 = bias_variable([num_labels])

    output = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

    print("Loss")
    with tf.name_scope("LOSS"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))

    print("Accuracy")
    with tf.name_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)), tf.float32))

    print("Optimizer")
    with tf.name_scope('Optimizer'):
        batch = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(1e-4, batch*batch_size, train_size, 0.95, staircase=True)
        grad = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=batch)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict={is_training: True})

        num_batches = int(train_size / batch_size)
        for epoch in range(max_epoch):
            for batch in range(num_batches):
                batch_offset = (batch * batch_size) % train_size
                batch_data = xtrain[batch_offset:(batch_offset+batch_size)]
                batch_labels = ytrain[batch_offset:(batch_offset+batch_size)]

                _, train_accuracy = sess.run([grad, accuracy], feed_dict={x:batch_data, y:batch_labels, keep_prob:0.5})

            print("Epoch:", (epoch+1), "/", max_epoch, "- Training accuracy:", train_accuracy)

        _, test_accuracy = sess.run([grad, accuracy], feed_dict={x: xtest, y: ytest, keep_prob:1.0})
        print("test accuracy for the stored model:", test_accuracy)
        saver.save(sess, '/dropout')

def get_weights():
    sess = tf.Session()
    new_saver = tf.train.import_meta_graph("model_var_drop.meta")
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    all_w = tf.get_collection('W_mu')
    f1 = open('var_dropout1.txt', 'ab')
    f2 = open('var_dropout2.txt', 'ab')
    count = 0
    for w in all_w:
        w_ = sess.run(w)
        w_ = w_.flatten()
        print(w_.shape)
        if count == 0:
            np.savetxt(f1, w_)
        else:
            np.savetxt(f2, w_)
        count += 1


def draw_hist():
    w1 = np.loadtxt('dropout1.txt')
    w1_mu = np.loadtxt('var_dropout1.txt')
    w1 = w1.reshape(w1.shape[0], 1)
    w1_mu = w1_mu.reshape(w1_mu.shape[0], 1)
    print(w1.shape, "", w1_mu.shape)

    w2 = np.loadtxt('dropout2.txt')
    w2_mu = np.loadtxt('var_dropout2.txt')
    w2 = w2.reshape(w2.shape[0], 1)
    w2_mu = w2_mu.reshape(w2_mu.shape[0], 1)
    print(w2.shape, "", w2_mu.shape)

    w3 = np.append(w1, w2, axis=0)
    w3_mu = np.append(w1_mu, w2_mu, axis=0)

    print(w3.shape, "", w3_mu.shape)

    plt.hist(w3, normed=True, label="Dropout",histtype = 'step')
    plt.hist(w3_mu, normed=True, label="Var. Dropout",histtype = 'step')
    plt.legend(loc='upper right')
    plt.show()


def draw_train_plot():
    train_dropout_acc = np.loadtxt('normal.txt')
    train_var_dro_acc = np.loadtxt('var_dropout.txt')

    plt.plot(train_dropout_acc, '-r', label='dropout')
    plt.plot(train_var_dro_acc, ':b', label='variational dropout')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    print("Variational dropout works first. Normal dropout will work after")
    run_variational_network()
    print("Starting normal dropout process")
    run_basic_network()
    #get_weights()
    #draw_hist()
    #draw_train_plot()
