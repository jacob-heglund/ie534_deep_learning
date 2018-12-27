"""
HW2: Implement and train a convolution neural network from scratch in Python for the MNIST dataset (no PyTorch). You should write your own code for convolutions (e.g., do not use SciPy's convolution function). The convolution network should have a single hidden layer with multiple channels. It should achieve 97-98% accuracy on the Test Set. For full credit, submit via Compass (1) the code and (2) a paragraph (in a PDF document) which states the Test Accuracy and briefly describes the implementation. Due September 14 at 5:00 PM.

Author: Jacob Heglund
"""
# Action Items
#TODO: get backprop working correctly (use signal.correlate2d with mode = 'valid' during this testing for faster epochs)
#TODO: get high test accuracy

##########################################################
# imports
import numpy as np
import h5py
import time
import copy
from random import randint
import matplotlib.pyplot as plt
from scipy import signal

##########################################################
# load MNIST data
def loadData(printShape):
    MNIST_data = h5py.File('MNISTdata.hdf5', 'r')

    # import training data
    # input imports as 60000, 784 array -> each row is a stretched out 28x28 image
    # output imports as 60000, 1 array -> 
    xTrain = np.float32(MNIST_data['x_train'][:])
    yTrain = np.int32(np.array(MNIST_data['y_train'][:,0]))

    # import testing data
    # input data consists of (10000, 784) data -> 10000, 28x28 images that have been stretched into vectors
    # output data consists of (10000, 1) data -> 10000, single digits 
    xTest = np.float32(MNIST_data['x_test'][:])
    yTest = np.int32(np.array(MNIST_data['y_test'][:,0]))
    MNIST_data.close()
    
    print("Data Imported\n")
    if printShape:
        print(
        "Data Size ", 
        "\nxTrain: ", np.shape(xTrain),
        "\nyTrain: ", np.shape(yTrain),
        "\nxTest: ", np.shape(xTest),
        "\nyTest: ", np.shape(yTest))

    return xTrain, yTrain, xTest, yTest

#####################################################
# define functions for the CNN
#CHECKED
def relu(z):
    return np.maximum(z, 0)
#CHECKED
def reluPrime(z):
    return (z > 0) * 1
#CHECKED
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis = 0)

#CHECKED
# don't touch this, its load-bearing garbage!
def myConvolve2DChannels(x, K):
    imageSide = np.shape(x)[0]

    numChannels = np.shape(K)[0]
    kx = np.shape(K)[1]
    ky = np.shape(K)[2]

    outputMatrixSizeX = imageSide - kx + 1
    outputMatrixSizeY = imageSide - ky + 1

    Z = np.zeros([numChannels, outputMatrixSizeX, outputMatrixSizeY])

    for channel in range(numChannels):
        for i in range(outputMatrixSizeX):
            for j in range(outputMatrixSizeY):
                # TODO: this will change depending on filter size, but works for a 3x3 filter
                imageSlice = x[i:i+kx, j:j+ky]
                # do elementwise multiplication
                intermediateMatrix = imageSlice * K[channel, :, :]
                total = np.sum(intermediateMatrix) 
                Z[channel, i, j] = total
            
    return Z

# this only works for "filter" size 26x26 (for backprop)
def myConvolve2D(x, K):
    imageSide = np.shape(x)[0]

    kx = np.shape(K)[0]
    ky = np.shape(K)[1]

    outputMatrixSizeX = imageSide - kx + 1
    outputMatrixSizeY = imageSide - ky + 1

    Z = np.zeros([outputMatrixSizeX, outputMatrixSizeY])

    for i in range(outputMatrixSizeX):
        for j in range(outputMatrixSizeY):
            imageSlice = x[i:i+kx, j:j+ky]
            # do elementwise multiplication
            intermediateMatrix = imageSlice * K
            total = np.sum(intermediateMatrix) 
            Z[i, j] = total
            
    return Z

#CHECKED
def indicator(z):
    # input: a scalar
    # output: vector with all entries zero except at index z which is equal to 1 
    zerosVec = np.zeros(numOutputs)
    zerosVec[z] = 1
    indicatorVector = np.reshape(zerosVec, [numOutputs, 1])
    return indicatorVector

#CHECKED
def cnnForward(x, y, model):
    forwardTerms = {}
    # convolution
    forwardTerms["Z"] = myConvolve2DChannels(x, model["K"])
    # activation function for 1st hidden layer
    forwardTerms["H"] = relu(forwardTerms["Z"])
    
    # can stretch out H into a vector in order to use W as with a vanilla NN (this is a pain as it changes the backprop update)
    #hVector = np.reshape(forwardTerms["H"], [26*26, 1])
    #for k in range(outputLayerSize):
    #    forwardTerms["U"][k] = np.matmul(model["W"][k], hVector) + model["b"][k]
    
    # fully connected layer  
    forwardTerms["U"] = np.zeros([numOutputs,1])
    for channel in range(numChannels):
        for k in range(numOutputs):
            forwardTerms["U"][k] = np.einsum('ij, ij', model["W"][k], forwardTerms["H"][channel])  + model["b"][k]

    forwardTerms["f"] = softmax(forwardTerms["U"])
    return forwardTerms

#CHECKED
def cnnBackward(x, y, model, forwardTerms):
    grads = {}

    # CHECKED
    drho_dU = -1 * (indicator(y) - forwardTerms["f"])

    # CHECKED    
    grads["b"] = drho_dU

    grads["W"] = np.zeros(np.shape(model["W"]))

    # delta is the same size as Z, the convolution of input x and filter K
    delta = np.zeros(np.shape(forwardTerms["Z"]))
    
    # CHECKED
    for channel in range(numChannels):
        for i in range(np.shape(delta)[0]):
            for j in range(np.shape(delta)[1]):
                wVector = np.reshape(model["W"][:, i, j], [numOutputs,1])
                delta[channel, i, j] = wVector.T.dot(drho_dU)
    
        # CHECKED
        # drho_dU[k] is a scalar, so regular multiplication works here
        for k in range(numOutputs):
            grads["W"][k] = drho_dU[k] * forwardTerms["H"][channel]

        # CHECKED
        grads["K"] = np.zeros(np.shape(model["K"]))
        # do elementwise multiplication here
        filterTerm = np.multiply(reluPrime(forwardTerms["Z"][channel]), delta[channel])
        grads["K"][channel] = myConvolve2D(x, filterTerm)

    return grads

#####################################################
# define model
np.random.seed(1)
numTrainingData = 60000
numTestingData = 10000

# number of inputs (MNIST pictures are 28x28 images)
inputLayerSize = 28*28
imageSideSize = 28

# filter properties
kx = 3
ky = 3
numChannels = 2

# number of outputs (classify images as digits from the set {0, 1, 2, ..., 9})
numOutputs = 10

model = {}
model["W"] = np.random.randn(numOutputs, imageSideSize-kx+1, imageSideSize-ky+1)
# model["W"] = np.random.randn(outputLayerSize, 1, 26*26)
model["K"] = np.random.randn(numChannels, kx, ky)
model["b"] = np.zeros([numOutputs])
model["b"] = np.reshape(model["b"], [numOutputs, 1])
#####################################################
# train the network
xTrain, yTrain, xTest, yTest = loadData(printShape = False)

numEpochs = 5

timeTrainingStart = time.time()

for epoch in range(numEpochs):
    timeEpochStart = time.time()
    totalCorrect = 0
    
    # Learning rate schedule
    alpha = 10**-1
    if (epoch > 5):
        alpha = 10**-2
    if (epoch > 10):
        alpha = 10**-3
    if (epoch > 15):
        alpha = 10**-4
    if (epoch > 20):
        alpha = 10**-5
    if (epoch > 25):
        alpha = 10**-5

    # create a randomly ordered list of numbers from 0-60000 for SGD
    numList = np.arange(0, numTrainingData, 1)
    np.random.shuffle(numList)

    # do SGD
    for n in range(numTrainingData):

        nRandom = numList[n] 
        
        # grab a random piece of data
        # cnn takes 28 x 28 image as input
        x = np.reshape(xTrain[nRandom, :], [28, 28])
        y = yTrain[nRandom]
        
        forwardTerms = cnnForward(x, y, model)
        prediction = np.argmax(forwardTerms["f"])
        if (prediction == y):
            totalCorrect += 1
        
        grads = cnnBackward(x, y, model, forwardTerms)
    
        model["b"] = model["b"] - alpha * grads["b"]
        model["K"] = model["K"] - alpha * grads["K"]
        model["W"] = model["W"] - alpha * grads["W"]

        if n % 1000 == 0:
            print("Training Step: ", n)

    timeEpochEnd = time.time()

    print("Epoch: ", str(epoch+1), 
    "--- Training Accuracy: ", str(totalCorrect/numTrainingData), 
    "--- Runtime (seconds): ", timeEpochEnd-timeEpochStart)

    if epoch % 1 == 0:
        totalCorrect = 0
        for n in range(numTestingData):
            # cnn takes 28 x 28 image as input
            x = np.reshape(xTest[n, :], [28, 28])
            y = yTest[n]
            if n % 1000 == 0:
                print("Testing Step: ", n)

            forwardTerms = cnnForward(x, y, model)
            prediction = np.argmax(forwardTerms["f"])

            if (prediction == y):
                totalCorrect += 1
        print("Epoch: ", str(epoch+1),
         "--- Testing Accuracy: ", str(totalCorrect/numTestingData))

timeTrainingEnd = time.time()
print('Runtime (seconds): ', timeTrainingEnd - timeTrainingStart)
