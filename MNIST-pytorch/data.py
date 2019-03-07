import numpy as np
import scipy.linalg
import os,time
import torch
import torchvision

import warp,util

# load MNIST data
def loadMNIST(opt,path):
	os.makedirs(path,exist_ok=True)
	trainDataset = torchvision.datasets.MNIST(path,train=True,download=True)
	testDataset = torchvision.datasets.MNIST(path,train=False,download=True)
	trainData,testData = {},{}
	trainData["image"] = torch.tensor([np.array(sample[0])/255.0 for sample in trainDataset],dtype=torch.float32)
	testData["image"] = torch.tensor([np.array(sample[0])/255.0 for sample in testDataset],dtype=torch.float32)
	trainData["label"] = torch.tensor([sample[1] for sample in trainDataset])
	testData["label"] = torch.tensor([sample[1] for sample in testDataset])
	return trainData,testData

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
	pInit = torch.from_numpy(pPert).cuda()
	return pInit

# make training batch
def makeBatch(opt,data):
	N = len(data["image"])
	randIdx = np.random.randint(N,size=[opt.batchSize])
	batch = {
		"image": data["image"][randIdx].cuda(),
		"label": data["label"][randIdx].cuda(),
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
		image = data["image"][idx].cuda()
		label = data["label"][idx].cuda()
		image.unsqueeze_(dim=1)
		# generate perturbation
		pInit = genPerturbations(opt)
		pInitMtrx = warp.vec2mtrx(opt,pInit)
		imagePert = warp.transformImage(opt,image,pInitMtrx)
		imageWarpAll = geometric(opt,image,pInit) if opt.netType=="IC-STN" else geometric(opt,imagePert)
		imageWarp = imageWarpAll[-1]
		output = classifier(opt,imageWarp)
		_,pred = output.max(dim=1)
		count += int((pred==label).sum().cpu().numpy())
		if opt.netType=="STN" or opt.netType=="IC-STN":
			imgPert = imagePert.detach().cpu().numpy()
			imgWarp = imageWarp.detach().cpu().numpy()
			for i in range(len(realIdx)):
				l = data["label"][idx[i]].item()
				if l not in warped[0]: warped[0][l] = []
				if l not in warped[1]: warped[1][l] = []
				warped[0][l].append(imgPert[i])
				warped[1][l].append(imgWarp[i])
	accuracy = float(count)/N
	if opt.netType=="STN" or opt.netType=="IC-STN":
		mean = [np.array([np.mean(warped[0][l],axis=0) for l in warped[0]]),
				np.array([np.mean(warped[1][l],axis=0) for l in warped[1]])]
		var = [np.array([np.var(warped[0][l],axis=0) for l in warped[0]]),
			   np.array([np.var(warped[1][l],axis=0) for l in warped[1]])]
	else: mean,var = None,None
	geometric.train()
	classifier.train()
	return accuracy,mean,var
