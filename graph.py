import numpy as np
import tensorflow as tf
import time

import data, warp


# auxiliary function for creating weight and bias
def createVariable(shape, stddev, zeroInit=False):
    if zeroInit:
        weight = tf.Variable(tf.zeros(shape), name="weight")
        bias = tf.Variable(tf.zeros([shape[-1]]), name="bias")
    else:
        weight = tf.Variable(tf.random_normal(shape, stddev=stddev), name="weight")
        bias = tf.Variable(tf.random_normal([shape[-1]], stddev=stddev), name="bias")
    return weight, bias


# build classification perceptron for MNIST
def buildPerceptron(image, stddev, params):
    imageVec = tf.reshape(image, [-1, params.H * params.W])
    with tf.variable_scope("fc1"):
        weight, bias = createVariable([params.H * params.W, 10], stddev)
        fc3 = tf.matmul(imageVec, weight) + bias
    return fc3


# build classification CNN for MNIST
def buildCNN(image, dimShape, stddev, params):
    [conv1dim] = dimShape
    conv1fcDim = (params.H - 8) * (params.W - 8) * conv1dim
    with tf.variable_scope("conv1"):
        weight, bias = createVariable([9, 9, 1, conv1dim], stddev)
        conv1 = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu1 = tf.nn.relu(conv1)
    relu1vec = tf.reshape(relu1, [-1, conv1fcDim])
    with tf.variable_scope("fc2"):
        weight, bias = createVariable([conv1fcDim, 10], stddev)
        fc2 = tf.matmul(relu1vec, weight) + bias
    return fc2


# build classification CNN for MNIST
def buildFullCNN(image, dimShape, stddev, params):
    [conv1dim, conv2dim, conv3dim, conv4dim, fc5dim] = dimShape
    conv4fcDim = ((params.H - 4) // 2 - 4) * ((params.W - 4) // 2 - 4) * conv4dim
    with tf.variable_scope("conv1"):
        weight, bias = createVariable([3, 3, 1, conv1dim], stddev)
        conv1 = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu1 = tf.nn.relu(conv1)
    with tf.variable_scope("conv2"):
        weight, bias = createVariable([3, 3, conv1dim, conv2dim], stddev)
        conv2 = tf.nn.conv2d(relu1, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu2 = tf.nn.relu(conv2)
        maxpool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    with tf.variable_scope("conv3"):
        weight, bias = createVariable([3, 3, conv2dim, conv3dim], stddev)
        conv3 = tf.nn.conv2d(maxpool2, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu3 = tf.nn.relu(conv3)
    with tf.variable_scope("conv4"):
        weight, bias = createVariable([3, 3, conv3dim, conv4dim], stddev)
        conv4 = tf.nn.conv2d(relu3, weight, strides=[1, 1, 1, 1], padding="VALID") + bias
        relu4 = tf.nn.relu(conv4)
    relu4vec = tf.reshape(relu4, [-1, conv4fcDim])
    with tf.variable_scope("fc5"):
        weight, bias = createVariable([conv4fcDim, fc5dim], stddev)
        fc5 = tf.matmul(relu4vec, weight) + bias
        relu5 = tf.nn.relu(fc5)
    with tf.variable_scope("fc6"):
        weight, bias = createVariable([fc5dim, 10], stddev)
        fc6 = tf.matmul(relu5, weight) + bias
    return fc6


# make image_summary from image batch
def makeImageSummary(tag, image, params):
    with tf.name_scope("imageSummary"):
        blockSize = params.visBlockSize
        imageSlice = tf.slice(image, [0, 0, 0, 0], [blockSize ** 2, -1, -1, -1])
        imageOne = tf.batch_to_space(imageSlice, crops=[[0, 0], [0, 0]], block_size=blockSize)
        imagePermute = tf.reshape(imageOne, [params.H, blockSize, params.W, blockSize, 1])
        imageTransp = tf.transpose(imagePermute, [1, 0, 3, 2, 4])
        imageBlocks = tf.reshape(imageTransp, [1, params.H * blockSize, params.W * blockSize, 1])
        tf.summary.image(tag, imageBlocks)


# set optimizer for different learning rates
def setOptimizer(loss, learningRate, params):
    varList = tf.global_variables()
    varListST = [v for v in varList if "ST" in v.name]
    varListOther = [v for v in varList if "ST" not in v.name]
    lrST, lrOther = tf.unstack(learningRate)
    gradients = tf.gradients(loss, varListST + varListOther)
    optimizerOther = tf.train.GradientDescentOptimizer(lrOther)
    gradientsOther = gradients[len(varListST):]
    trainStepOther = optimizerOther.apply_gradients(zip(gradientsOther, varListOther))
    if len(varListST) > 0:
        optimizerST = tf.train.GradientDescentOptimizer(lrST)
        gradientsST = gradients[:len(varListST)]
        trainStepST = optimizerST.apply_gradients(zip(gradientsST, varListST))
        trainStep = tf.group(trainStepST, trainStepOther)
    else:
        trainStep = trainStepOther
    return trainStep
