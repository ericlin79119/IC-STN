import numpy as np
import time, os, sys
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("type", metavar="TYPE", help="type of network", choices=["CNN", "STN", "cSTN", "ICSTN"])
parser.add_argument("--group", default="0", help="name for group")
parser.add_argument("--model", default="test", help="name for model instance")
parser.add_argument("--recurN", type=int, default=4, help="number of recurrent transformations (for IC-STN)")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for classification network")
parser.add_argument("--lrST", type=float, default=1e-4, help="learning rate for geometric predictor")
parser.add_argument("--batchSize", type=int, default=200, help="batch size for SGD")
parser.add_argument("--maxIter", type=int, default=100000, help="maximum number of training iterations")
parser.add_argument("--warpType", metavar="WARPTYPE", help="type of warp function on images", default="homography",
                    choices=["translation", "similarity", "affine", "homography"])
parser.add_argument("--resume", type=int, default=0, help="resume from iteration number")
parser.add_argument("--gpu", type=int, default=0, help="ID of GPU device (if there are multiple)")
args = parser.parse_args()

import tensorflow as tf
import data, graph, graphST, warp
from params import Params

print("=======================================================")
print("train.py (training on MNIST)")
print("=======================================================")

# load data
print("loading MNIST dataset...")
trainData, validData, testData = data.loadMNIST("data/MNIST.npz")

# set parameters
print("setting configurations...")
params = Params(args)

# create directories for model output
suffix = args.group
if not os.path.exists("models_{0}".format(suffix)): os.mkdir("models_{0}".format(suffix))
if not os.path.exists("models_{0}/interm".format(suffix)): os.mkdir("models_{0}/interm".format(suffix))
if not os.path.exists("models_{0}/final".format(suffix)): os.mkdir("models_{0}/final".format(suffix))
saveFname = args.model

print("training model {0}...".format(saveFname))
print("------------------------------------------")
print("warpScale: (pert) {0} (trans) {1}".format(params.warpScale["pert"], params.warpScale["trans"]))
print("warpType: {0}".format(params.warpType))
print("batchSize: {0}".format(params.batchSize))
print("GPU device: {0}".format(args.gpu))
print("------------------------------------------")

tf.reset_default_graph()
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
# build graph
with tf.device(params.GPUdevice):
    # generate training data on the fly
    imageRawBatch = tf.placeholder(tf.float32, shape=[None, 28, 28], name="image")
    pInitBatch = data.genPerturbations(params)
    pInitMtrxBatch = warp.vec2mtrxBatch(pInitBatch, params)
    ImBatch = data.imageWarpIm(imageRawBatch, pInitMtrxBatch, params, name=None)
    # build network
    if args.type == "CNN":
        outputBatch = graph.buildFullCNN(ImBatch, [3, 6, 9, 12, 48], 0.1, params)
    elif args.type == "STN":
        ImWarpBatch, pBatch = graphST.ST_depth4_CCFF(ImBatch, pInitBatch, 1, [4, 8, 48], 0.01, params)
        outputBatch = graph.buildCNN(ImWarpBatch, [3], 0.03, params)
    elif args.type == "cSTN":
        ImWarpBatch, pBatch = graphST.cST_depth4_CCFF(imageRawBatch, pInitBatch, 1, [4, 8, 48], 0.01, params)
        outputBatch = graph.buildCNN(ImWarpBatch, [3], 0.03, params)
    elif args.type == "ICSTN":
        ImWarpBatch, pBatch = graphST.cSTrecur_depth4_CCFF(imageRawBatch, pInitBatch, args.recurN, [4, 8, 48], 0.01,
                                                           params)
        outputBatch = graph.buildCNN(ImWarpBatch, [3], 0.03, params)
    # define loss/optimizer/summaries
    imageSummaries = tf.summary.merge_all()
    labelBatch = tf.placeholder(tf.float32, shape=[None, 10], name="label")
    softmaxLoss = tf.nn.softmax_cross_entropy_with_logits(logits=outputBatch, labels=labelBatch)
    loss = tf.reduce_mean(softmaxLoss)
    lossSummary = tf.summary.scalar("training loss", loss)
    learningRate = tf.placeholder(tf.float32, shape=[2])
    trainStep = graph.setOptimizer(loss, learningRate, params)
    softmax = tf.nn.softmax(outputBatch)
    prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labelBatch, 1))

print("starting backpropagation...")
trainN = len(trainData["image"])
timeStart = time.time()
# define some more summaries
tfSaver, tfSaverInterm, tfSaverFinal = tf.train.Saver(max_to_keep=10), \
                                       tf.train.Saver(max_to_keep=10), \
                                       tf.train.Saver()
testErrorPH = tf.placeholder(tf.float32, shape=[])
testErrorSummary = tf.summary.scalar("test error", testErrorPH)
tfSummaryWriter = tf.summary.FileWriter("summary_{1}/{0}".format(saveFname, suffix))
resumeIterN = 0
maxIterN = 100000
with tf.Session(config=tfConfig) as sess:
    if resumeIterN == 0:
        sess.run(tf.global_variables_initializer())
    else:
        tfSaver.restore(sess, "models_{2}/{0}_it{1}k.ckpt".format(saveFname, resumeIterN // 1000, suffix))
        print("resuming from iteration {0}...".format(resumeIterN))
    tfSummaryWriter.add_graph(sess.graph)
    # training loop
    for i in range(resumeIterN, maxIterN):
        currLearningRate = params.baseLRST, params.baseLR  # this can be modified to be scheduled learning rates
        randIdx = np.random.randint(trainN, size=[params.batchSize])
        trainBatch = {
            imageRawBatch: trainData["image"][randIdx],
            labelBatch: trainData["label"][randIdx],
            learningRate: currLearningRate
        }
        # run one step
        _, trainBatchLoss, summary = sess.run([trainStep, loss, lossSummary], feed_dict=trainBatch)
        if (i + 1) % 20 == 0:
            tfSummaryWriter.add_summary(summary, i + 1)
        if (i + 1) % 100 == 0:
            print("it. {0}/{1} (lr={5:.2e},{4:.2e}), loss={2:.4f}, time={3:.4f}"
                  .format(i + 1, maxIterN, trainBatchLoss, time.time() - timeStart, currLearningRate[0],
                          currLearningRate[1]))
        if (i + 1) % 2000 == 0:
            # update image summaries
            if imageSummaries is not None:
                summary = sess.run(imageSummaries, feed_dict=trainBatch)
                tfSummaryWriter.add_summary(summary, i + 1)
            # evaluate on validation and test sets
            testAccuracy = data.evaluate(testData, imageRawBatch, pInitBatch, labelBatch, prediction, sess, params)
            testError = (1 - testAccuracy) * 100
            summary = sess.run(testErrorSummary, feed_dict={testErrorPH: testError})
            tfSummaryWriter.add_summary(summary, i + 1)
            # save model
            savePath = tfSaver.save(sess, "models_{2}/{0}_it{1}k.ckpt".format(saveFname, (i + 1) // 1000, suffix))
            print("model saved: {0}".format(savePath))
        if (i + 1) % 10000 == 0:
            # save intermediate model
            tfSaverInterm.save(sess, "models_{2}/interm/{0}_it{1}k.ckpt".format(saveFname, (i + 1) // 1000, suffix))
    # save final model
    tfSaverFinal.save(sess, "models_{1}/final/{0}.ckpt".format(saveFname, suffix))
print("======= backpropagation done =======")
