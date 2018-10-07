import numpy as np
import torch
import time
import data,warp,util

# build classification network
class FullCNN(torch.nn.Module):
	def __init__(self,opt):
		super(FullCNN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[3,3],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(3),torch.nn.ReLU(True),
			conv2Layer(6),torch.nn.ReLU(True),maxpoolLayer(),
			conv2Layer(9),torch.nn.ReLU(True),
			conv2Layer(12),torch.nn.ReLU(True)
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.labelN)
		)
		initialize(opt,self,opt.stdC)
	def forward(self,opt,image):
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		output = feat
		return output

# build classification network
class CNN(torch.nn.Module):
	def __init__(self,opt):
		super(CNN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[9,9],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(3),torch.nn.ReLU(True)
		)
		self.inDim *= 20**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(opt.labelN)
		)
		initialize(opt,self,opt.stdC)
	def forward(self,opt,image):
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		output = feat
		return output

# an identity class to skip geometric predictors
class Identity(torch.nn.Module):
	def __init__(self): super(Identity,self).__init__()
	def forward(self,opt,feat): return [feat]

# flatten feature map to 1d vector
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

# build Spatial Transformer Network
class STN(torch.nn.Module):
	def __init__(self,opt):
		super(STN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[7,7],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(4),torch.nn.ReLU(True),
			conv2Layer(8),torch.nn.ReLU(True),maxpoolLayer()
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.warpDim)
		)
		initialize(opt,self,opt.stdGP,last0=True)
	def forward(self,opt,image):
		imageWarpAll = [image]
		feat = image
		feat = self.conv2Layers(feat).view(opt.batchSize,-1)
		feat = self.linearLayers(feat)
		p = feat
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarpAll

# build compositional STN
class CSTN(torch.nn.Module):
	def __init__(self, opt):
		super(CSTN, self).__init__()
		
		def conv2Layer(inDim, outDim):
			conv = torch.nn.Conv2d(inDim,outDim,kernel_size=[9,9],stride=1,padding=0)
			return conv
		
		def linearLayer(inDim, outDim):
			fc = torch.nn.Linear(inDim,outDim)
			return fc

		def build_c_stn(num_stn, stn_list):
			assert(num_stn == 2 or num_stn == 4)
			#structure for c-stn-2, MNIST
			if num_stn == 2:
				for i in range(num_stn):
					stn = torch.nn.Sequential(
						conv2Layer(1,4), torch.nn.ReLU(True),
						Flatten(),
						linearLayer(1600, opt.warpDim)
					)
					stn_list.append(stn)
			
			#structure for c-stn-4, MNIST
			if num_stn == 4:
				for i in range(num_stn):
					stn = torch.nn.Sequential(
						Flatten(),
						linearLayer(28*28*1, opt.warpDim)
					)
					stn_list.append(stn)
		
		self.c_stn_x = torch.nn.ModuleList()
		build_c_stn(opt.stnN, self.c_stn_x)
		initialize_cstn(opt,self,opt.stdGP,last0=True)

	def forward(self, opt, image, p):
		imageWarpAll = []
		for l in range(opt.stnN):
			pMtrx = warp.vec2mtrx(opt,p)
			imageWarp = warp.transformImage(opt,image,pMtrx)
			imageWarpAll.append(imageWarp)
			feat = imageWarp
			feat = self.c_stn_x[l](feat)
			dp = feat
			p = warp.compose(opt,p,dp)
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarpAll

	# def forward(self, image):
	# 	imageWarpAll = []
	# 	for l in range(NOT_DEFINED):
	# 		feat = image
	# 		feat = self.c_stn_x[l](feat)
	# 		imageWarpAll.append(image)
	# 	return imageWarpAll



# build Inverse Compositional STN
class ICSTN(torch.nn.Module):
	def __init__(self,opt):
		super(ICSTN,self).__init__()
		self.inDim = 1
		def conv2Layer(outDim):
			conv = torch.nn.Conv2d(self.inDim,outDim,kernel_size=[7,7],stride=1,padding=0)
			self.inDim = outDim
			return conv
		def linearLayer(outDim):
			fc = torch.nn.Linear(self.inDim,outDim)
			self.inDim = outDim
			return fc
		def maxpoolLayer(): return torch.nn.MaxPool2d([2,2],stride=2)
		self.conv2Layers = torch.nn.Sequential(
			conv2Layer(4),torch.nn.ReLU(True),
			conv2Layer(8),torch.nn.ReLU(True),maxpoolLayer()
		)
		self.inDim *= 8**2
		self.linearLayers = torch.nn.Sequential(
			linearLayer(48),torch.nn.ReLU(True),
			linearLayer(opt.warpDim)
		)
		initialize(opt,self,opt.stdGP,last0=True)
	def forward(self,opt,image,p):
		imageWarpAll = []
		for l in range(opt.warpN):
			pMtrx = warp.vec2mtrx(opt,p)
			imageWarp = warp.transformImage(opt,image,pMtrx)
			imageWarpAll.append(imageWarp)
			feat = imageWarp
			feat = self.conv2Layers(feat).view(opt.batchSize,-1)
			feat = self.linearLayers(feat)
			dp = feat
			p = warp.compose(opt,p,dp)
		pMtrx = warp.vec2mtrx(opt,p)
		imageWarp = warp.transformImage(opt,image,pMtrx)
		imageWarpAll.append(imageWarp)
		return imageWarpAll

# initialize weights/biases
def initialize(opt,model,stddev,last0=False):
	for m in model.conv2Layers:
		if isinstance(m,torch.nn.Conv2d):
			m.weight.data.normal_(0,stddev)
			m.bias.data.normal_(0,stddev)
	for m in model.linearLayers:
		if isinstance(m,torch.nn.Linear):
			if last0 and m is model.linearLayers[-1]:
				m.weight.data.zero_()
				m.bias.data.zero_()
			else:
				m.weight.data.normal_(0,stddev)
				m.bias.data.normal_(0,stddev)

def initialize_cstn(opt,model,stddev,last0=False):
	for l in model.c_stn_x:
		for m in l:
			if isinstance(m,torch.nn.Conv2d):
				# print('conv2d init')
				m.weight.data.normal_(0,stddev)
				m.bias.data.normal_(0,stddev)
			if isinstance(m,torch.nn.Linear):
				# print('linear init')
				if last0 and m is l[-1]:
					m.weight.data.zero_()
					m.bias.data.zero_()
				else:
					m.weight.data.normal_(0,stddev)
					m.bias.data.normal_(0,stddev)