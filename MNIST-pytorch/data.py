import numpy as np
import scipy.linalg
import os,time
import torch

import warp,util

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
		trainData["label"] = np.argmax(mnist.train.labels.astype(np.float32),axis=1)
		validData["label"] = np.argmax(mnist.validation.labels.astype(np.float32),axis=1)
		testData["label"] = np.argmax(mnist.test.labels.astype(np.float32),axis=1)
		os.makedirs(os.path.dirname(fname))
		np.savez(fname,train=trainData,valid=validData,test=testData)
		os.system("rm -rf MNIST_data")
	MNIST = np.load(fname)
	trainData = MNIST["train"].item()
	validData = MNIST["valid"].item()
	testData = MNIST["test"].item()
	return trainData,validData,testData

# generate training batch
def genPerturbations(opt):
	X = np.tile(opt.canon4pts[:,0],[opt.batchSize,1])
	Y = np.tile(opt.canon4pts[:,1],[opt.batchSize,1])
	O = np.zeros([opt.batchSize,4],dtype=np.float32)
	I = np.ones([opt.batchSize,4],dtype=np.float32)
	dX = np.random.randn(opt.batchSize,4)*opt.pertScale \
		+np.random.randn(opt.batchSize,1)*opt.transScale
	dY = np.random.randn(opt.batchSize,4)*opt.pertScale \
		+np.random.randn(opt.batchSize,1)*opt.transScale
	dX,dY = dX.astype(np.float32),dY.astype(np.float32)
	# fit warp parameters to generated displacements
	if opt.warpType=="homography":
		A = np.concatenate([np.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
							np.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],axis=1)
		b = np.expand_dims(np.concatenate([X+dX,Y+dY],axis=1),axis=-1)
		pPert = np.matmul(np.linalg.inv(A),b).squeeze()
		pPert -= np.array([1,0,0,0,1,0,0,0])
	else:
		if opt.warpType=="translation":
			J = np.concatenate([np.stack([I,O],axis=-1),
								np.stack([O,I],axis=-1)],axis=1)
		if opt.warpType=="similarity":
			J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
								np.stack([-Y,X,O,I],axis=-1)],axis=1)
		if opt.warpType=="affine":
			J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
								np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
		dXY = np.expand_dims(np.concatenate([dX,dY],axis=1),axis=-1)
		Jtransp = np.transpose(J,axes=[0,2,1])
		pPert = np.matmul(np.linalg.inv(np.matmul(Jtransp,J)),np.matmul(Jtransp,dXY)).squeeze()
	pInit = util.toTorch(pPert)
	return pInit

def genIdentity(opt):
	X = np.tile(opt.canon4pts[:,0],[opt.batchSize,1])
	Y = np.tile(opt.canon4pts[:,1],[opt.batchSize,1])
	O = np.zeros([opt.batchSize,4],dtype=np.float32)
	I = np.ones([opt.batchSize,4],dtype=np.float32)
	#set dx and dy to 0
	dX = np.zeros([opt.batchSize,4],dtype=np.float32)
	dY = np.zeros([opt.batchSize,4],dtype=np.float32)
	dX,dY = dX.astype(np.float32),dY.astype(np.float32)
	# fit warp parameters to generated displacements
	if opt.warpType=="homography":
		A = np.concatenate([np.stack([X,Y,I,O,O,O,-X*(X+dX),-Y*(X+dX)],axis=-1),
							np.stack([O,O,O,X,Y,I,-X*(Y+dY),-Y*(Y+dY)],axis=-1)],axis=1)
		b = np.expand_dims(np.concatenate([X+dX,Y+dY],axis=1),axis=-1)
		pPert = np.matmul(np.linalg.inv(A),b).squeeze()
		pPert -= np.array([1,0,0,0,1,0,0,0])
	else:
		if opt.warpType=="translation":
			J = np.concatenate([np.stack([I,O],axis=-1),
								np.stack([O,I],axis=-1)],axis=1)
		if opt.warpType=="similarity":
			J = np.concatenate([np.stack([X,Y,I,O],axis=-1),
								np.stack([-Y,X,O,I],axis=-1)],axis=1)
		if opt.warpType=="affine":
			J = np.concatenate([np.stack([X,Y,I,O,O,O],axis=-1),
								np.stack([O,O,O,X,Y,I],axis=-1)],axis=1)
		dXY = np.expand_dims(np.concatenate([dX,dY],axis=1),axis=-1)
		Jtransp = np.transpose(J,axes=[0,2,1])
		pPert = np.matmul(np.linalg.inv(np.matmul(Jtransp,J)),np.matmul(Jtransp,dXY)).squeeze()
	pInit = util.toTorch(pPert)
	# print('pinit shape:', pInit.shape)
	# print('pinit:', pInit)
	return pInit

# make training batch
def makeBatch(opt,data):
	N = len(data["image"])
	randIdx = np.random.randint(N,size=[opt.batchSize])
	batch = {
		"image": util.toTorch(data["image"][randIdx]),
		"label": util.toTorch(data["label"][randIdx]),
	}
	return batch

# evaluation on test set
def evalTest(opt,data,geometric,classifier):
	geometric.eval()
	classifier.eval()
	N = len(data["image"])
	batchN = int(np.ceil(N/opt.batchSize))
	warped = [{},{}]
	count = 0
	for b in range(batchN):
		# use some dummy data (0) as batch filler if necessary
		if b!=batchN-1:
			realIdx = np.arange(opt.batchSize*b,opt.batchSize*(b+1))
		else:
			realIdx = np.arange(opt.batchSize*b,N)
		idx = np.zeros([opt.batchSize],dtype=int)
		idx[:len(realIdx)] = realIdx
		# make training batch
		image = util.toTorch(data["image"][idx])
		label = util.toTorch(data["label"][idx])
		image.data.unsqueeze_(dim=1)
		# generate perturbation
		pInit = genPerturbations(opt)
		pInitMtrx = warp.vec2mtrx(opt,pInit)
		imagePert = warp.transformImage(opt,image,pInitMtrx)
		imageWarpAll = geometric(opt,image,pInit) if opt.netType=="IC-STN" or opt.netType=="C-STN" or opt.netType=="DeSTN" else geometric(opt,imagePert)
		imageWarp = imageWarpAll[-1]
		output = classifier(opt,imageWarp)
		_,pred = output.max(dim=1)
		count += int(util.toNumpy((pred==label).sum()))
		if opt.netType=="STN" or opt.netType=="IC-STN" or opt.netType=="C-STN" or opt.netType=="DeSTN":
			imgPert = util.toNumpy(imagePert)
			imgWarp = util.toNumpy(imageWarp)
			for i in range(len(realIdx)):
				l = data["label"][idx[i]]
				if l not in warped[0]: warped[0][l] = []
				if l not in warped[1]: warped[1][l] = []
				warped[0][l].append(imgPert[i])
				warped[1][l].append(imgWarp[i])
	accuracy = float(count)/N
	if opt.netType=="STN" or opt.netType=="IC-STN" or opt.netType=="C-STN" or opt.netType=="DeSTN":
		mean = [np.array([np.mean(warped[0][l],axis=0) for l in warped[0]]),
				np.array([np.mean(warped[1][l],axis=0) for l in warped[1]])]
		var = [np.array([np.var(warped[0][l],axis=0) for l in warped[0]]),
			   np.array([np.var(warped[1][l],axis=0) for l in warped[1]])]
	else: mean,var = None,None
	geometric.train()
	classifier.train()
	return accuracy,mean,var

# evaluation on valid set
def evalValid(opt,data,geometric,classifier):
	return evalTest(opt,data,geometric,classifier)