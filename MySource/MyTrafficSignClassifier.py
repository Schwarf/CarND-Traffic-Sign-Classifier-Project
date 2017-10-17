'''
Created on Oct 14, 2017

@author: andre
'''
import pickle 
from sklearn.utils import shuffle
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import time
import datetime
import MyDNN
from dask.array.chunk import keepdims_wrapper
import csv

trainingFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project2/Dataset/traffic-signs-data/train.p"
validationFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project2/Dataset/traffic-signs-data/valid.p"
testingFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project2/Dataset/traffic-signs-data/test.p"

signNamesFile = "D:/Andreas/Programming/Python/UdacitySelfDrivingCar/Term1Projects/Project2/CarND-Traffic-Sign-Classifier-Project/signnames.csv"

def ReadData(file):
    with open(file, mode='rb') as f:
        data = pickle.load(f)
        features, labels = data['features'], data['labels']
    assert(len(features) == len(labels))
    return features, labels
    
def ShowSampleData(features, labels):
    labelNames = {}
    with open(signNamesFile, mode='r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            if row[0].isdigit():
                labelNames[int(row[0])] = row[1]
    plt.figure(figsize=(12, 16.5), dpi =80)
    for signID, signName in labelNames.items():
        plt.subplot(11, 4, 1+signID)
        selectedFeatures = features[labels == signID]
        randomize = np.random.randint(0,100)
        plt.imshow(selectedFeatures[randomize,:,:,:])
        plt.tight_layout()
        plt.title(signName, fontsize = 8)
    #plt.show()


def DataExploration(y, name):
    result = np.unique(y, return_counts =True)
    assert(len(result[1]) == 43)
    plt.bar(result[0], result[1], 0.8, color = 'green')
    plt.title('Distribution of classes in '+name, fontsize=10)
    plt.tight_layout()
    #plt.show()

def ApplyCannyEdgeDetection(image, lowerThreshold=50, upperThreshold=150):
    return cv2.Canny(image, lowerThreshold, upperThreshold)


def ApplyGaussianSmoothing(image, kernelSize =5):
    return cv2.GaussianBlur(image,(kernelSize, kernelSize),0)

def ConvertHSVImageToGrayColorSpace(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return ConvertBGRImageToGrayColorSpace(image1)

def ConvertBGRImageToGrayColorSpace(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def ConvertImageToHSVSpace(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def ConvertImageToYUVSpace(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    
def ExtractY(image):
    y,_,_ = cv2.split(ConvertImageToYUVSpace(image))    
    return y

def ExtractV(image):
    _,_,v = cv2.split(ConvertImageToHSVSpace(image))    
    return v

def ImageFeaturesPreProcessing(inputImages, YUV = True, imageWeight = 1.5, noiseWeight = -0.8):
    numberOfImages = inputImages.shape[0] 
    imageWidth = inputImages.shape[1] 
    imageHeight = inputImages.shape[2] 
    resultingImages = np.ndarray((numberOfImages, imageWidth, imageHeight, 1), dtype=np.uint8)
    for index, image in enumerate(inputImages):
        if(YUV):
            '''
            Use YUV color space
            '''
            image = ExtractY(image)
        else:
            '''
            Use HSV color space
            '''
            image = ExtractV(image)
        image = cv2.equalizeHist(image)
        image = np.expand_dims(image, axis=2)
        resultingImages[index] = image
        
    return resultingImages


def NormalizeData(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean)/std




def ApplyImageRotation(image, angle):
    imageCenterWidth =image.shape[0]//2
    imageCenterHeight =image.shape[1]//2
    rotationMatrix = cv2.getRotationMatrix2D((imageCenterWidth, imageCenterHeight),angle,1)
    return cv2.warpAffine(image,rotationMatrix,(image.shape[0],image.shape[1]))   

def ApplyImageTranslation(image, translation):
    translationMatrix = np.float32([[1,0,translation[0]],[0,1,translation[1]]])
    return cv2.warpAffine(image,translationMatrix,(image.shape[0],image.shape[1]))
    
def TransformImage(image):
    angle = np.random.uniform(-15,15)
    translation = np.random.randint(-2,2,2)
    rotatedImage = ApplyImageRotation(image, angle)
    translatedImage = ApplyImageTranslation(rotatedImage, translation)
    return translatedImage
      

def GenerateNewData(numberOfDataSets, underlyingImages, assignedLabel):
    generatedImageSamples = np.random.randint(len(underlyingImages), size = numberOfDataSets)
    newImages = []
    newLabels = [assignedLabel]*numberOfDataSets
    for index in generatedImageSamples:
        newImages.append(TransformImage(underlyingImages[index]))
    return newImages, newLabels



def DataAugmentation(features, labels, requiredInstancesPerSign):
    trafficSigns, trafficSignCounts = np.unique(labels, return_counts=True)
    SignsAndCounts = dict(zip(trafficSigns, trafficSignCounts))
    print( "Relative frequency required for each class in whole set: "+ repr(requiredInstancesPerSign)) 
    for signID, signCount in SignsAndCounts.items():
        if(signCount > requiredInstancesPerSign):           
            continue
        numberOfInstancesToGenerate = requiredInstancesPerSign - signCount
        featuresForNewInstanceGeneration = features[labels == signID]
        newImages, newLabels = GenerateNewData(numberOfInstancesToGenerate, featuresForNewInstanceGeneration,signID)
        assert(len(newLabels) == len(newImages))
        newImages = np.asarray(newImages, dtype=features.dtype)    
        newLabels = np.asarray(newLabels, dtype=labels.dtype)
        features = np.append(features, newImages, axis=0)
        labels = np.append(labels, newLabels, axis=0)
    return features, labels

BATCH_SIZE = 128
EPOCHS = 20
learningRate = 0.001

'''
Reading the data
'''
featuresTraining, labelsTraining = ReadData(trainingFile)
featuresValidation, labelsValidation = ReadData(validationFile)
featuresTesting, labelsTesting = ReadData(testingFile)
'''
Visualize the data
'''

#ShowSampleData(featuresTraining, labelsTraining)
#DataExploration(labelsTraining, "Training labels")
#DataExploration(labelsValidation, "Validation labels")
#DataExploration(labelsTesting, "Testing labels")

'''
Data augmentation
'''
featuresTraining, labelsTraining = DataAugmentation(featuresTraining, labelsTraining, requiredInstancesPerSign =1000)

'''
Visualize augmented data
'''
DataExploration(labelsTraining, "Training labels after data augmentation")

'''
Preprocess all images
'''
featuresTraining = ImageFeaturesPreProcessing(featuresTraining)
featuresValidation = ImageFeaturesPreProcessing(featuresValidation)
featuresTesting = ImageFeaturesPreProcessing(featuresTesting)

'''
Normalize data
'''
featuresTraining = NormalizeData(featuresTraining)
featuresValidation = NormalizeData(featuresValidation)
featuresTesting = NormalizeData(featuresTesting)


''' Shuffle training data: training data is ordered'''
featuresTraining, labelsTraining = shuffle(featuresTraining, labelsTraining)



x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)

one_hot_y = tf.one_hot(y, 43)



logits = MyDNN.LeNet(x, keep_prob)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)    
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def AccuracyAnalysis(epoch, trainingAccuracy ,currentAccuracy, formerAccuracy, maximumAccuracy):
        print("EPOCH {} ...".format(epoch + 1))
        print("Training Accuracy [%] = {:.5f}".format(trainingAccuracy*100.0))
        print("Validation Accuracy [%] = {:.5f}".format(currentAccuracy*100.0))
        if(currentAccuracy > maximumAccuracy):
            maximumAccuracy = currentAccuracy

        if(formerAccuracy > 0.0):
            print("Relative change of validation accuracy [%] = {:.5f}".format((currentAccuracy - formerAccuracy)/formerAccuracy*100.0))
        if(maximumAccuracy > 0.0):
            print("Current validation accuracy in terms of maximum validation accuracy = {:.5f}".format(currentAccuracy/maximumAccuracy))
        print()
        formerAccuracy = currentAccuracy
        return formerAccuracy, maximumAccuracy
        

def evaluate(featuresData, labelsData):
    numberOfDataSets = len(featuresData)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, numberOfDataSets, BATCH_SIZE):
        batch_x, batch_y = featuresData[offset:offset + BATCH_SIZE], labelsData[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / numberOfDataSets


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    numberOfDataSets = len(featuresTraining)
    
    
    print("Training...")
    print()
    formerValidationAccuracy = 0.0
    maximumValidationAccuracy = 0.0
    for epoch in range(EPOCHS):
        featuresTraining, labelsTraining = shuffle(featuresTraining, labelsTraining)
        for offset in range(0, numberOfDataSets, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = featuresTraining[offset:end], labelsTraining[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        currentValidationAccuracy = evaluate(featuresValidation, labelsValidation)
        currentTrainingAccuracy = evaluate(featuresTraining, labelsTraining)
        
        formerValidationAccuracy, maximumValidationAccuracy = AccuracyAnalysis(epoch, currentTrainingAccuracy, currentValidationAccuracy, formerValidationAccuracy, maximumValidationAccuracy)
        
    saver.save(sess, './Project2/CarND-Traffic-Sign-Classifier-Project/TrafficSignClassifierModel/Model')
    print("Model saved")

with open("TrafficSignClassifierLeNetData.csv", "a") as myfile:
    writer = csv.writer(myfile, delimiter=',')
    writer.writerow([datetime.datetime.now(), learningRate, BATCH_SIZE, EPOCHS, currentValidationAccuracy*100.0, maximumValidationAccuracy*100.0])




#Run testing
'''
with tf.Session() as sess:
    saver.restore(sess, './Project2/CarND-Traffic-Sign-Classifier-Project/TrafficSignClassifierModel')
    test_accuracy = evaluate(featuresTesting, labelsTesting)
    print("Test Accuracy = {:.5f}".format(test_accuracy))
'''



    