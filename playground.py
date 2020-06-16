import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

const = tf.constant(2.0, name = "const")

b = tf.placeholder(tf.float32, [None, 1], name = 'b')
c = tf.Variable(1.0, name = 'c')

d = tf.add(b, c, name = 'd')
e = tf.add(c, const, 'e')
a = tf.multiply(d, e, name = 'a')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # the whole new axis shit:
    # np.arange -> an array (  e.g. [1, 2, 3, 4]  )
    # BUT we want
    a_out = sess.run(a, feed_dict = { b: np.arange(0, 10)[:, np.newaxis] })
    # a_out = sess.run(a, feed_dict = { b: np.arange(0, 10) })

    print("Variable a is {}".format(a_out))
