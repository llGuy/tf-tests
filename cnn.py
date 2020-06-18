import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

learning_rate = 0.0001
epochs = 10
batch_size = 50

x = tf.placeholder(tf.float32, [None, 784])

# In order to do max pooling, we need to provide a 4D piece of data
# Format is [i, j, k, l]
# i = number of training samples
# j = the height of the image
# k = the width
# l = number of channels (RGBA channels)
# In this case, we don't know when the x is (because we provided None), so it'll change
# depending on that
x_shaped = tf.reshape(x, [-1, 28, 28, 1])

y = tf.placeholder(tf.float32, [None, 10])

def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # Setup the filter input shape for tf.nn.conv_2d
    # Has to come in the following order:
    # [ filter_height, filter-width, in_channels, out_channels]
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    # Initialise the weights and bias for the filter (randomly)
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev = 0.03),
            name = name + "_W")
    bias = tf.Variable(tf.truncated_normal([num_filters]), name = name + "_b")

    # Setup the convolutional layer operation
    # [input data (duh), weights, the strides parameter, padding type]
    # With the padding type of "SAME" - you get the following that happens to the w / h
    # out_height = ceil(float(in_height) / float(strides[1]))
    # out_width = ceil(float(in_width) / float(strides[2]))
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding = 'SAME')

    # Add the bias
    out_layer += bias

    # Apply the ReLU non=linear activation
    out_layer = tf.nn.relu(out_layer)

    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize = ksize, strides = strides, padding = 'SAME')

    return out_layer

layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name = 'layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name = 'layer2')

# Setup fully connected layer and weights and biases for the fully connected layer
flattened = tf.reshape(layer2, [-1, 7 * 7 * 64])
wd1 = tf.Variable(tf.truncated_normal([7 * 7 *  64, 1000], stddev = 0.03), name = 'wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev = 0.01), name = 'bd1')

dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, 10], stddev = 0.03), name = 'wd2')
bd2 = tf.Variable(tf.truncated_normal([10], stddev = 0.01), name = 'bd2')

dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = dense_layer2, labels = y))

optimiser = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

# Accuracy assessment
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    total_batch = int(len(mnist.train.labels) / batch_size)

    for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict = {x: batch_x, y: batch_y})

            avg_cost += c / total_batch

        test_acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels})

        print("Epoch: ", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "test accuracy: {:.3f}".format(test_acc))


    print("\n Training complete!")
    print(sess.run(accuracy, feed_dict = {x: mnist.test.images, y: mnist.test.labels}))