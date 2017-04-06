import numpy as np
import scipy.linalg
import os,time
import tensorflow as tf

import warp

# load MNIST data
def loadMNIST(fname):
	if not os.path.exists(fname):
		# download and preprocess MNIST dataset
		from tensorflow.examples.tutorials.mnist import input_data
		mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
		trainData,validData,testData = {},{},{}
		trainData["image"] = mnist.train.images.reshape([-1,28,28]).astype(np.float32)
		validData["image"] = mnist.validation.images.reshape([-1,28,28]).astype(np.float32)
		testData["image"] = mnist.test.images.reshape([-1,28,28]).astype(np.float32)
		trainData["label"] = mnist.train.labels.astype(np.float32)
		validData["label"] = mnist.validation.labels.astype(np.float32)
		testData["label"] = mnist.test.labels.astype(np.float32)
		os.makedirs(os.path.dirname(fname))
		np.savez(fname,train=trainData,valid=validData,test=testData)
	MNIST = np.load(fname)
	trainData = MNIST["train"].item()
	validData = MNIST["valid"].item()
	testData = MNIST["test"].item()
	return trainData,validData,testData

# generate training batch
def genPerturbations(params):
	with tf.name_scope("genPerturbations"):
		X = np.tile(params.canon4pts[:,0],[params.batchSize,1])
		Y = np.tile(params.canon4pts[:,1],[params.batchSize,1])
		dX = tf.random_normal([params.batchSize,4])*params.warpScale["pert"] \
			+tf.random_normal([params.batchSize,1])*params.warpScale["trans"]
		dY = tf.random_normal([params.batchSize,4])*params.warpScale["pert"] \
			+tf.random_normal([params.batchSize,1])*params.warpScale["trans"]
		O = np.zeros([params.batchSize,4],dtype=np.float32)
		I = np.ones([params.batchSize,4],dtype=np.float32)
		# fit warp parameters to generated displacements
		if params.warpType=="affine":
			J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
								np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
			dXY = tf.expand_dims(tf.concat(1,[dX,dY]),-1)
			dpBatch = tf.matrix_solve_ls(J,dXY)
		elif params.warpType=="homography":
			A = tf.concat([tf.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
							 tf.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)], 1)
			b = tf.expand_dims(tf.concat([X+dX,Y+dY], 1),-1)
			dpBatch = tf.matrix_solve_ls(A,b)
			dpBatch -= tf.to_float(tf.reshape([1,0,0,0,1,0,0,0],[1,8,1]))
		dpBatch = tf.reduce_sum(dpBatch,reduction_indices=-1)
		dpMtrxBatch = warp.vec2mtrxBatch(dpBatch,params)
	return dpBatch

# evaluation on validation/test sets
def evaluate(data,imageRawBatch,pInitBatch,labelBatch,prediction,sess,params):
	dataN = len(data["image"])
	batchN = int(np.ceil(float(dataN)/params.batchSize))
	accurateCount = 0
	for b in range(batchN):
		# use some dummy data (0) as batch filler if necesaary
		if b!=batchN-1: realIdx = np.arange(params.batchSize*b,params.batchSize*(b+1))
		else: realIdx = np.arange(params.batchSize*b,dataN)
		idx = np.zeros([params.batchSize],dtype=int)
		idx[:len(realIdx)] = realIdx
		batch = {
			imageRawBatch: data["image"][idx],
			labelBatch: data["label"][idx],
		}
		predictionBatch = sess.run(prediction,feed_dict=batch)
		accurateCount += predictionBatch[:len(realIdx)].sum()
	accuracy = float(accurateCount)/dataN
	return accuracy

# generate batch of warped images from batch (bilinear interpolation)
def imageWarpIm(imageBatch,pMtrxBatch,params,name=None):
	with tf.name_scope("ImWarp"):
		imageBatch = tf.expand_dims(imageBatch,-1)
		batchSize = tf.shape(imageBatch)[0]
		imageH,imageW = params.H,params.H
		H,W = params.H,params.W
		warpGTmtrxBatch = tf.tile(tf.expand_dims(params.warpGTmtrx,0),[batchSize,1,1])
		transMtrxBatch = tf.matmul(warpGTmtrxBatch,pMtrxBatch)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
		XYhom = tf.transpose(tf.stack([X.reshape([-1]),Y.reshape([-1]),np.ones([X.size])],axis=1))
		XYhomBatch = tf.tile(tf.expand_dims(XYhom,0),[batchSize,1,1])
		XYwarpHomBatch = tf.matmul(transMtrxBatch,tf.to_float(XYhomBatch))
		XwarpHom,YwarpHom,ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
		Xwarp = tf.reshape(XwarpHom/ZwarpHom,[batchSize,H,W])
		Ywarp = tf.reshape(YwarpHom/ZwarpHom,[batchSize,H,W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		imageIdx = tf.tile(tf.reshape(tf.range(batchSize),[batchSize,1,1]),[1,H,W])
		imageVec = tf.reshape(imageBatch,[-1,tf.shape(imageBatch)[3]])
		imageVecOutside = tf.concat([imageVec,tf.zeros([1,tf.shape(imageBatch)[3]])], 0)
		idxUL = (imageIdx*imageH+YfloorInt)*imageW+XfloorInt
		idxUR = (imageIdx*imageH+YfloorInt)*imageW+XceilInt
		idxBL = (imageIdx*imageH+YceilInt)*imageW+XfloorInt
		idxBR = (imageIdx*imageH+YceilInt)*imageW+XceilInt
		idxOutside = tf.fill([batchSize,H,W],batchSize*imageH*imageW)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<imageW)&(Yint>=0)&(Yint<imageH)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[batchSize,H,W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[batchSize,H,W,1])
		ImUL = tf.to_float(tf.gather(imageVecOutside,idxUL))*(1-Xratio)*(1-Yratio)
		ImUR = tf.to_float(tf.gather(imageVecOutside,idxUR))*(Xratio)*(1-Yratio)
		ImBL = tf.to_float(tf.gather(imageVecOutside,idxBL))*(1-Xratio)*(Yratio)
		ImBR = tf.to_float(tf.gather(imageVecOutside,idxBR))*(Xratio)*(Yratio)
		ImWarpBatch = ImUL+ImUR+ImBL+ImBR
		ImWarpBatch = tf.identity(ImWarpBatch,name=name)
	return ImWarpBatch

# generate batch of warped images from batch (bilinear interpolation)
def ImWarpIm(ImBatch,pMtrxBatch,params,name=None):
	with tf.name_scope("ImWarp"):
		batchSize = tf.shape(ImBatch)[0]
		H,W = params.H,params.W
		warpGTmtrxBatch = tf.tile(tf.expand_dims(params.warpGTmtrx,0),[batchSize,1,1])
		transMtrxBatch = tf.matmul(warpGTmtrxBatch,pMtrxBatch)
		# warp the canonical coordinates
		X,Y = np.meshgrid(np.linspace(-1,1,W),np.linspace(-1,1,H))
		XYhom = tf.transpose(tf.stack([X.reshape([-1]),Y.reshape([-1]),np.ones([X.size])],axis=1))
		XYhomBatch = tf.tile(tf.expand_dims(XYhom,0),[batchSize,1,1])
		XYwarpHomBatch = tf.matmul(transMtrxBatch,tf.to_float(XYhomBatch))
		XwarpHom,YwarpHom,ZwarpHom = tf.split(XYwarpHomBatch, 3, 1)
		Xwarp = tf.reshape(XwarpHom/ZwarpHom,[batchSize,H,W])
		Ywarp = tf.reshape(YwarpHom/ZwarpHom,[batchSize,H,W])
		# get the integer sampling coordinates
		Xfloor,Xceil = tf.floor(Xwarp),tf.ceil(Xwarp)
		Yfloor,Yceil = tf.floor(Ywarp),tf.ceil(Ywarp)
		XfloorInt,XceilInt = tf.to_int32(Xfloor),tf.to_int32(Xceil)
		YfloorInt,YceilInt = tf.to_int32(Yfloor),tf.to_int32(Yceil)
		ImIdx = tf.tile(tf.reshape(tf.range(batchSize),[batchSize,1,1]),[1,H,W])
		ImVecBatch = tf.reshape(ImBatch,[-1,tf.shape(ImBatch)[3]])
		ImVecBatchOutside = tf.concat([ImVecBatch,tf.zeros([1,tf.shape(ImBatch)[3]])], 0)
		idxUL = (ImIdx*H+YfloorInt)*W+XfloorInt
		idxUR = (ImIdx*H+YfloorInt)*W+XceilInt
		idxBL = (ImIdx*H+YceilInt)*W+XfloorInt
		idxBR = (ImIdx*H+YceilInt)*W+XceilInt
		idxOutside = tf.fill([batchSize,H,W],batchSize*H*W)
		def insideIm(Xint,Yint):
			return (Xint>=0)&(Xint<W)&(Yint>=0)&(Yint<H)
		idxUL = tf.where(insideIm(XfloorInt,YfloorInt),idxUL,idxOutside)
		idxUR = tf.where(insideIm(XceilInt,YfloorInt),idxUR,idxOutside)
		idxBL = tf.where(insideIm(XfloorInt,YceilInt),idxBL,idxOutside)
		idxBR = tf.where(insideIm(XceilInt,YceilInt),idxBR,idxOutside)
		# bilinear interpolation
		Xratio = tf.reshape(Xwarp-Xfloor,[batchSize,H,W,1])
		Yratio = tf.reshape(Ywarp-Yfloor,[batchSize,H,W,1])
		ImUL = tf.to_float(tf.gather(ImVecBatchOutside,idxUL))*(1-Xratio)*(1-Yratio)
		ImUR = tf.to_float(tf.gather(ImVecBatchOutside,idxUR))*(Xratio)*(1-Yratio)
		ImBL = tf.to_float(tf.gather(ImVecBatchOutside,idxBL))*(1-Xratio)*(Yratio)
		ImBR = tf.to_float(tf.gather(ImVecBatchOutside,idxBR))*(Xratio)*(Yratio)
		ImWarpBatch = ImUL+ImUR+ImBL+ImBR
		ImWarpBatch = tf.identity(ImWarpBatch,name=name)
	return ImWarpBatch


