'''
Created on Oct 14, 2017

@author: andre
'''

from tensorflow.contrib.layers import flatten
import tensorflow as tf

def BiasInitialization(dim):
    return tf.Variable(tf.zeros([dim]))


def MyDNN(x, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    weightsLayer1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev =sigma))
    biasLayer1 = BiasInitialization(6)
    paddingLayer1 = 'VALID'
    stridesLayer1 = [1,1,1,1]
    layer1 =  tf.nn.conv2d(x,weightsLayer1, stridesLayer1, paddingLayer1) + biasLayer1
    
    print(tf.shape(layer1))
    # TODO: Activation.
    activation1 = tf.nn.relu(layer1)
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    filterPooling1 = [1,2,2,1]
    stridesPooling1 = [1,2,2,1]
    paddingPooling1 = 'VALID'
    maxPooling1 = tf.nn.max_pool(activation1, filterPooling1, stridesPooling1, paddingPooling1)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weightsLayer2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev =sigma))
    biasLayer2 = BiasInitialization(16)
    paddingLayer2 = 'VALID'
    stridesLayer2 = [1,1,1,1]
    layer2 =  tf.nn.conv2d(maxPooling1,weightsLayer2, stridesLayer2, paddingLayer2) + biasLayer2
    
    # TODO: Activation.
    activation2 = tf.nn.relu(layer2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    filterPooling2 = [1,2,2,1]
    stridesPooling2 = [1,2,2,1]
    paddingPooling2 = 'VALID'
    maxPooling2 = tf.nn.max_pool(activation2, filterPooling2, stridesPooling2, paddingPooling2)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten1 = flatten(maxPooling2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    filter3 = [400,120]
    weightsLayer3 = tf.Variable(tf.truncated_normal(filter3, mean = mu, stddev =sigma))
    biasLayer3 = BiasInitialization(120)
    layer3 = tf.matmul(flatten1, weightsLayer3) + biasLayer3
    
    
    # TODO: Activation.
    activation3 = tf.nn.relu(layer3)
    activation3 = tf.nn.dropout(activation3, keep_prob)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    filter4 = [120,84]
    weightsLayer4 = tf.Variable(tf.truncated_normal(filter4, mean = mu, stddev =sigma))
    biasLayer4 = BiasInitialization(84)
    layer4 = tf.matmul(activation3, weightsLayer4) + biasLayer4
    
   
    # TODO: Activation.
    activation4 = tf.nn.relu(layer4)
    activation4 = tf.nn.dropout(activation4, keep_prob)
   
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    filter5 = [84,43]
    weightsLayer5 = tf.Variable(tf.truncated_normal(filter5, mean = mu, stddev =sigma))
    biasLayer5 = BiasInitialization(43)
    layer5 = tf.matmul(activation4, weightsLayer5) + biasLayer5
    
    return layer5