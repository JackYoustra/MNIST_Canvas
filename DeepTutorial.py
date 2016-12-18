'''
Created on Dec 13, 2016

@author: jack
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

def weight_variable(shape):
    # going to have ReLU (floored direct proportional regressions) neurons,
    # so include some noise to prevent 0 gradients
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# and also want to initialize with a slightly positive initial bias to avoid,
# "dead" (unactivated) neurons
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# do plain vanilla 2D convolution - output is the same size as the input with zero padding - control spacial size of output volumes
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def trainModel():
    global x
    global y_
    global sess
    global y_conv
    global keep_prob
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sess = tf.InteractiveSession() #can let you interleave operations when building computation graph, helpful with IPython
    
    # input into computation graph at a later time - what a placeholder means.
    # Shape is optional but helps TF catch inconsistent tensor shape errors - v helpful
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    
    # define weights and biases for model
    # weights is 1 per input neuron to each output neuron, so tensor from each input (784) to each output (10)
    # bias is one per output
    W = tf.Variable(tf.zeros([784,10])) 
    b = tf.Variable(tf.zeros([10]))
    
    # initialize all variables (take the argument tensors and do the "setting" of the variables at this step)
    sess.run(tf.global_variables_initializer())
    
    y = tf.matmul(x,W) + b #easy regression, multiply by weights and add bias
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_)) # find loss function (applies softmax over unnormalized and then normalized it)
    
    # pull single step of gradient descent
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    for i in range(1000):
        batch = mnist.train.next_batch(100) # get a batch of 100 examples to run on
        train_step.run(feed_dict={x: batch[0], y_: batch[1]}) # run training on the batch, repeat 1k times with different batches
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    # uses convolution followed by max pooling
    W_conv1 = weight_variable([5, 5, 1, 32]) # 32 features per 5x5 patch (1 input channel, 32 output channels)
    b_conv1 = bias_variable([32]) # one bias vector per output channel
    
    # reshape x to a 4D tensor, 2nd and 3rd are width, height, 4th is color channels
    x_image = tf.reshape(x, [-1,28,28,1])

    # convolve
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    
    # max pool, reduces to 14x14 image (half)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # second convolutional layer. 64 features for each 5x5 patch on the 14x14 image, thus doing a reduction
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2) # reduces to 7x7 (half)
    
    # add fully-connected layer when left with 7x7 image with 64 features with 1024 neurons as output for processing on entire image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # apply dropout layer, have the probability of neuron dropping out be a parameter (placeholder)
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # add readout layer to read out results - fully connected
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    #normal multiply and bias add
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_)) # normal cross entropy
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) # ADAM instead of gradient descent: separate learning rate per parameter, momentum like simulated annealing (converging schedule) I THINK
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) #gradient normal stuff
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for i in range(200000): #jesus big train. On google's number of repetitions isn't accurate, but boost it by x10 and works
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    

# take prediction image and return number result
def predict(image):
    # now run on my own system, entirely self-created
    myTestSet = np.ndarray(shape=(1, 784))
    myTestLabels = np.full((1, 10), 0.0)
    myTestLabels[0, 4] = 1.0 # equal to true for 4
    
    im = image.copy()
    im.thumbnail((28, 28), Image.NEAREST)
    im.save("tmp.png")
    pix = np.array(im)
    myTestSet = pix * (1.0 / 255.0)
    myTestSet = np.reshape(myTestSet, (1, 784))
    myTestSet = np.absolute(np.subtract(np.full((1, 784), 1.0), myTestSet))
        
    accuracy = tf.argmax(y_conv, 1)
    print(sess.run(accuracy, feed_dict={x: myTestSet, y_: myTestLabels, keep_prob: 1.0}))
    
if __name__ == "__main__":
    trainModel()
    