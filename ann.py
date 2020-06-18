import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.5
epochs = 10
batch_size = 100

# The images are of dimensions 28 * 28
x = tf.placeholder(tf.float32, [None, 28 * 28])

# The output (which digit is it, but in an array - 0 for all the ones that are not, and 1 for the one that is correct)
y = tf.placeholder(tf.float32, [None, 10])

# 3.0) A Neural Network Example (https://adventuresinmachinelearning.com/python-tensorflow-tutorial/)
# 2.5) The Notation (https://adventuresinmachinelearning.com/neural-networks-tutorial/)

# Weights connecting layer 1 (input layer) to layer 2 (hidden layer)
w1 = tf.Variable(tf.random_normal([784, 300], stddev = 0.03), name = 'w1')
b1 = tf.Variable(tf.random_normal([300]), name = 'b1')

# Weights connecting layer 2 (hidden layer) to layer 3 (output layer)
w2 = tf.Variable(tf.random_normal([300, 10], stddev = 0.03), name = 'w2')
b2 = tf.Variable(tf.random_normal([10]), name = 'b2')

hidden_out = tf.add(tf.matmul(x, w1), b1)
# Activation function
# If the value is less than 0, then simply clip to 0
hidden_out = tf.nn.relu(hidden_out)

# Calculate the hidden layer output
y_ = tf.add(tf.matmul(hidden_out, w2), b2)

# Softmax squishes the values so that all the values in the tensors add up to 1
y_ = tf.nn.softmax(y_)

# Here we calculate the cross entropy which we want to minise with the optimiser
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis = 1))

optimiser = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Just for ONE image (after having trained the neural net)
print(mnist.train.labels[0])

# image_data = tf.placeholder(tf.float32, [1, 28 * 28])

image_hidden_in = tf.add(tf.matmul(x, w1), b1)
image_hidden_in = tf.nn.relu(hidden_out)

image_hidden_out = tf.add(tf.matmul(image_hidden_in, w2), b2)
image_hidden_out = tf.nn.softmax(image_hidden_out)

with tf.Session() as sess:
    sess.run(init_op)

    total_batch = int(len(mnist.train.labels) / batch_size)
    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost = ", "{:.3f}".format(avg_cost))
    # print(sess.run(accuracy, feed_dict = {x: mnist.train.images[0], y: mnist.train.labels[0]}))

    result = sess.run(image_hidden_out, feed_dict = {x: [mnist.train.images[0]], y: [mnist.train.labels[0]] })

    print(result)
    
    print(mnist.train.labels[0])

    sess.close()

