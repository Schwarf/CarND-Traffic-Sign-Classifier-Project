'''
Created on Oct 14, 2017

@author: andre
'''

from tensorflow.contrib.layers import flatten
import tensorflow as tf


def SlimDNN(input, keep_prob):
    mu = 0
    sigma = 0.1

    
    #print ("input shape ", input.shape)
    weightsLayer0 = tf.Variable(tf.truncated_normal([11, 11, 1, 16], mean = mu, stddev =sigma))
    biasLayer0 = BiasInitialization(16)
    paddingLayer0 = 'SAME'
    stridesLayer0 = [1,1,1,1]
    layer0 =  tf.nn.conv2d(input,weightsLayer0, stridesLayer0, paddingLayer0) + biasLayer0
    # TODO: Activation.
    #print ("Layer1 ", layer1.shape)
    activation0 = tf.nn.relu(layer0)
    #activation0 = tf.nn.dropout(activation0, keep_prob)
    
    #print ("input shape ", input.shape)
    weightsLayer1 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], mean = mu, stddev =sigma))
    biasLayer1 = BiasInitialization(32)
    paddingLayer1 = 'SAME'
    stridesLayer1 = [1,2,2,1]
    layer1 =  tf.nn.conv2d(activation0,weightsLayer1, stridesLayer1, paddingLayer1) + biasLayer1
    # TODO: Activation.
    #print ("Layer1 ", layer1.shape)
    activation1 = tf.nn.relu(layer1)
    activation1 = tf.nn.dropout(activation1, keep_prob)

    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    filterPooling1 = [1,2,2,1]
    stridesPooling1 = [1,2,2,1]
    paddingPooling1 = 'SAME'
    maxPooling1 = tf.nn.max_pool(activation1, filterPooling1, stridesPooling1, paddingPooling1)
    
    
    
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weightsLayer2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean = mu, stddev =sigma))
    biasLayer2 = BiasInitialization(64)
    paddingLayer2 = 'SAME'
    stridesLayer2 = [1,2,2,1]
    layer2 =  tf.nn.conv2d(maxPooling1,weightsLayer2, stridesLayer2, paddingLayer2) + biasLayer2
    #print ("Layer2 ", layer2.shape)
    
    # TODO: Activation.
    activation2 = tf.nn.relu(layer2)
    activation2 = tf.nn.dropout(activation2, keep_prob)
    
    
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    filterPooling2 = [1,2,2,1]
    stridesPooling2 = [1,2,2,1]
    paddingPooling2 = 'SAME'
    maxPooling2 = tf.nn.max_pool(activation2, filterPooling2, stridesPooling2, paddingPooling2)
    #print ("Maxpooling2 ", maxPooling2.shape)
    
    
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten1 = flatten(maxPooling2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    
    filter3 = [256,43]
    weightsLayer3 = tf.Variable(tf.truncated_normal(filter3, mean = mu, stddev =sigma))
    biasLayer3 = BiasInitialization(43)
    layer3 = tf.matmul(flatten1, weightsLayer3) + biasLayer3
    #print ("layer3 ", layer3.shape)
    
    return layer3




def MyDNN(input, keep_prob):
    mu = 0
    sigma = 0.1

    
    #print ("input shape ", input.shape)
    weightsLayer0 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], mean = mu, stddev =sigma))
    biasLayer0 = BiasInitialization(32)
    paddingLayer0 = 'VALID'
    stridesLayer0 = [1,1,1,1]
    layer0 =  tf.nn.conv2d(input,weightsLayer0, stridesLayer0, paddingLayer0) + biasLayer0
    # TODO: Activation.
    #print ("Layer1 ", layer1.shape)
    activation0 = tf.nn.relu(layer0)
    
    #print ("input shape ", input.shape)
    weightsLayer1 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], mean = mu, stddev =sigma))
    biasLayer1 = BiasInitialization(64)
    paddingLayer1 = 'VALID'
    stridesLayer1 = [1,1,1,1]
    layer1 =  tf.nn.conv2d(activation0,weightsLayer1, stridesLayer1, paddingLayer1) + biasLayer1
    # TODO: Activation.
    #print ("Layer1 ", layer1.shape)
    activation1 = tf.nn.relu(layer1)


    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    filterPooling1 = [1,2,2,1]
    stridesPooling1 = [1,2,2,1]
    paddingPooling1 = 'VALID'
    maxPooling1 = tf.nn.max_pool(activation1, filterPooling1, stridesPooling1, paddingPooling1)
    #print ("Maxpooling1 ", maxPooling1.shape)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weightsLayer2 = tf.Variable(tf.truncated_normal([5, 5, 64, 128], mean = mu, stddev =sigma))
    biasLayer2 = BiasInitialization(128)
    paddingLayer2 = 'VALID'
    stridesLayer2 = [1,1,1,1]
    layer2 =  tf.nn.conv2d(maxPooling1,weightsLayer2, stridesLayer2, paddingLayer2) + biasLayer2
    #print ("Layer2 ", layer2.shape)
    
    # TODO: Activation.
    activation2 = tf.nn.relu(layer2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    filterPooling2 = [1,2,2,1]
    stridesPooling2 = [1,2,2,1]
    paddingPooling2 = 'VALID'
    maxPooling2 = tf.nn.max_pool(activation2, filterPooling2, stridesPooling2, paddingPooling2)
    #print ("Maxpooling2 ", maxPooling2.shape)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten1 = flatten(maxPooling2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    
    filter3 = [2048,200]
    weightsLayer3 = tf.Variable(tf.truncated_normal(filter3, mean = mu, stddev =sigma))
    biasLayer3 = BiasInitialization(200)
    layer3 = tf.matmul(flatten1, weightsLayer3) + biasLayer3
    #print ("layer3 ", layer3.shape)
    
    
    # TODO: Activation.
    activation3 = tf.nn.relu(layer3)
    activation3 = tf.nn.dropout(activation3, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    filter4 = [200,100]
    weightsLayer4 = tf.Variable(tf.truncated_normal(filter4, mean = mu, stddev =sigma))
    biasLayer4 = BiasInitialization(100)
    layer4 = tf.matmul(activation3, weightsLayer4) + biasLayer4
    #print ("layer4 ", layer4.shape)
    
   
    # TODO: Activation.
    activation4 = tf.nn.relu(layer4)
    activation4 = tf.nn.dropout(activation4, keep_prob)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    filter5 = [100,43]
    weightsLayer5 = tf.Variable(tf.truncated_normal(filter5, mean = mu, stddev =sigma))
    biasLayer5 = BiasInitialization(43)
    layer5 = tf.matmul(activation4, weightsLayer5) + biasLayer5
    #print ("layer5 ", layer5.shape)
    
    
    return layer5

def BiasInitialization(dim):
    return tf.Variable(tf.zeros([dim]))


def ConvoltionalLayer(input,strides, filterWidth, filterHeight, inputDepth, outputDepth, mu =0, sigma =0.1):
    convolutionalWeights = tf.Variable(tf.truncated_normal([filterWidth, filterHeight, inputDepth, outputDepth], mean = mu, stddev =sigma))
    convolutionalBias = tf.Variable(tf.zeros(outputDepth))
    return tf.nn.conv2d(input, convolutionalWeights, strides=strides, padding='VALID') + convolutionalBias

def ReluLayer(input,strides, filterWidth, filterHeight, inputDepth, outputDepth, mu =0, sigma =0.1):
    layer = ConvoltionalLayer(input,strides,filterWidth, filterHeight, inputDepth, outputDepth, mu, sigma)
    return tf.nn.relu(layer)

def FullyConnectedLayer(input, inputDepth, outputDepth, mu =0, sigma =0.1):
    filter = [inputDepth, outputDepth]
    weights = tf.Variable(tf.truncated_normal(filter, mean = mu, stddev =sigma))
    bias = tf.Variable(tf.zeros(outputDepth))
    return tf.matmul(input, weights) + bias


def MaxPooling(input, filterSize, strideSize, padding):
    poolingFilter = [1,filterSize, filterSize,1]
    poolingStrides = [1,strideSize,strideSize,1]
    return tf.nn.max_pool(input, poolingFilter, poolingStrides, padding)

def LeNet(input, keep_prob):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    
    #print ("input shape ", input.shape)
    weightsLayer1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], mean = mu, stddev =sigma))
    biasLayer1 = BiasInitialization(6)
    paddingLayer1 = 'VALID'
    stridesLayer1 = [1,1,1,1]
    layer1 =  tf.nn.conv2d(input,weightsLayer1, stridesLayer1, paddingLayer1) + biasLayer1
    # TODO: Activation.
    #print ("Layer1 ", layer1.shape)
    activation1 = tf.nn.relu(layer1)
    
    
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    filterPooling1 = [1,2,2,1]
    stridesPooling1 = [1,2,2,1]
    paddingPooling1 = 'VALID'
    maxPooling1 = tf.nn.max_pool(activation1, filterPooling1, stridesPooling1, paddingPooling1)
    #print ("Maxpooling1 ", maxPooling1.shape)
    
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    weightsLayer2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean = mu, stddev =sigma))
    biasLayer2 = BiasInitialization(16)
    paddingLayer2 = 'VALID'
    stridesLayer2 = [1,1,1,1]
    layer2 =  tf.nn.conv2d(maxPooling1,weightsLayer2, stridesLayer2, paddingLayer2) + biasLayer2
    #print ("Layer2 ", layer2.shape)
    
    # TODO: Activation.
    activation2 = tf.nn.relu(layer2)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    filterPooling2 = [1,2,2,1]
    stridesPooling2 = [1,2,2,1]
    paddingPooling2 = 'VALID'
    maxPooling2 = tf.nn.max_pool(activation2, filterPooling2, stridesPooling2, paddingPooling2)
    #print ("Maxpooling2 ", maxPooling2.shape)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flatten1 = flatten(maxPooling2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    
    filter3 = [400,120]
    weightsLayer3 = tf.Variable(tf.truncated_normal(filter3, mean = mu, stddev =sigma))
    biasLayer3 = BiasInitialization(120)
    layer3 = tf.matmul(flatten1, weightsLayer3) + biasLayer3
    #print ("layer3 ", layer3.shape)
    
    
    # TODO: Activation.
    activation3 = tf.nn.relu(layer3)
    activation3 = tf.nn.dropout(activation3, keep_prob)
    
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    filter4 = [120,84]
    weightsLayer4 = tf.Variable(tf.truncated_normal(filter4, mean = mu, stddev =sigma))
    biasLayer4 = BiasInitialization(84)
    layer4 = tf.matmul(activation3, weightsLayer4) + biasLayer4
    #print ("layer4 ", layer4.shape)
    
   
    # TODO: Activation.
    activation4 = tf.nn.relu(layer4)
    activation4 = tf.nn.dropout(activation4, keep_prob)
    
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    filter5 = [84,43]
    weightsLayer5 = tf.Variable(tf.truncated_normal(filter5, mean = mu, stddev =sigma))
    biasLayer5 = BiasInitialization(43)
    layer5 = tf.matmul(activation4, weightsLayer5) + biasLayer5
    #print ("layer5 ", layer5.shape)

    return layer5