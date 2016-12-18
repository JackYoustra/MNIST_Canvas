'''
Created on Nov 29, 2016

@author: jack
'''

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) #directory for storing input data
    # placeholders = parameters
    mnistNumberPlaceholder = tf.placeholder(tf.float32, [None, 784]) # things to fill in as the image in the dataflow graph
    # model parameters (weights, biases, etc) - usually live in variables, a modifiable tensor that acts as a variable in the dataflow graph
    # initialize variable tensor with zeros b/c going to train so doesn't matter
    weights = tf.Variable(tf.zeros([784, 10])) # shape because 784 weights (pixel weights) over 10 different classes (the digits)
    biases = tf.Variable(tf.zeros([10]))
    
    outputs = tf.nn.softmax(tf.matmul(mnistNumberPlaceholder, weights) + biases)
    
    # to train, use cross entropy function: (tex) H_{y'}(y) = -\sum_i y'_i \log(y_i)
    outputs_ = tf.placeholder(tf.float32, [None, 10]) # input correct answers in this placeholder
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(outputs_ * tf.log(outputs), reduction_indices=[1])) #cross entropy: (tex) -\sum y'\log(y). Mathematically unstable, consider switching
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outputs, outputs_)) # numerically stable version of the above because cross entropy is calculated on the raw outputs and then averaged
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy) # apply backpropagation to find derivatives and employ gradient descent with learning rate of 0.5
    
    #initialize variables step
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init) # run initialization step
    for i in range(1000): #train 1k times
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={mnistNumberPlaceholder: batch_xs, outputs_: batch_ys})
    
    #test
    # tf.argmax IS DA REAL MVP, gives index of highest entry along axis, making it reeeealy easy to find most likely label and correct label
    correct_prediction = tf.equal(tf.argmax(outputs, 1), tf.argmax(outputs_, 1))
    # we now have a list of booleans. To find percentage correct, cast to floating point and then take mean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # now ask for accuracy
    print(sess.run(accuracy, feed_dict={mnistNumberPlaceholder: mnist.test.images, outputs_: mnist.test.labels}))
    
    #results:
    # with non-numerically stable function, 0.9189
    # with numerically stable function, 0.9069
    
    # now run on my own system, entirely self-created
    myTestSet = np.ndarray(shape=(1, 784))
    myTestLabels = np.full((1, 10), 0.0)
    myTestLabels[0, 4] = 1.0 # equal to true for 4
    
    im = Image.open("four mid reduce nn.jpg") #works best with nearest neighbor
    pix = np.array(im)
    myTestSet = pix * (1.0 / 255.0)
    myTestSet = np.reshape(myTestSet, (1, 784))
    myTestSet = np.absolute(np.subtract(np.full((1, 784), 1.0), myTestSet))
        
    print(str(0) + "; test: " + str(myTestLabels[0]))
    
    print(sess.run(accuracy, feed_dict={mnistNumberPlaceholder: myTestSet, outputs_: myTestLabels}))
    
    