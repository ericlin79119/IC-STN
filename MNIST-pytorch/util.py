import numpy as np
import scipy.misc
import torch
import os
import termcolor
import visdom

def mkdir(path):
	if not os.path.exists(path): os.mkdir(path)
def imread(fname):
	return scipy.misc.imread(fname)/255.0
def imsave(fname,array):
	scipy.misc.toimage(array,cmin=0.0,cmax=1.0).save(fname)

def toTorch(nparray):
	tensor = torch.from_numpy(nparray).cuda()
	return torch.autograd.Variable(tensor,requires_grad=False)
def toNumpy(cudavar):
	return cudavar.data.cpu().numpy()

# convert to colored strings
def toRed(content): return termcolor.colored(content,"red",attrs=["bold"])
def toGreen(content): return termcolor.colored(content,"green",attrs=["bold"])
def toBlue(content): return termcolor.colored(content,"blue",attrs=["bold"])
def toCyan(content): return termcolor.colored(content,"cyan",attrs=["bold"])
def toYellow(content): return termcolor.colored(content,"yellow",attrs=["bold"])
def toMagenta(content): return termcolor.colored(content,"magenta",attrs=["bold"])

# restore model
def restoreModel(opt,geometric,classifier,it):
	geometric.load_state_dict(torch.load("models_{0}/{1}_it{2}_GP.npy".format(opt.group,opt.model,it)))
	classifier.load_state_dict(torch.load("models_{0}/{1}_it{2}_C.npy".format(opt.group,opt.model,it)))
# save model
def saveModel(opt,geometric,classifier,it):
	torch.save(geometric.state_dict(),"models_{0}/{1}_it{2}_GP.npy".format(opt.group,opt.model,it))
	torch.save(classifier.state_dict(),"models_{0}/{1}_it{2}_C.npy".format(opt.group,opt.model,it))

class Visdom():
	def __init__(self,opt):
		self.vis = visdom.Visdom(port=opt.port)
		self.trainLossInit = True
		self.validLossInit = True
		self.testLossInit = True
		self.meanVarInit = True
	
	def tileImages(self,opt,images,H,W,HN,WN):
		assert(len(images)==HN*WN)
		images = images.reshape([HN,WN,-1,H,W])
		images = [list(i) for i in images]
		imageBlocks = np.concatenate([np.concatenate(row,axis=2) for row in images],axis=1)
		return imageBlocks
	
	def trainLoss(self,opt,it,loss):
		loss = float(toNumpy(loss))
		if self.trainLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_trainloss".format(opt.model),
						  opts={ "title": "{0} (TRAIN_loss)".format(opt.model) })
			self.trainLossInit = False
		else: self.vis.line(Y=np.array([loss]),X=np.array([it]),win=opt.model+"_trainloss",update="append")
	
	def testLoss(self,opt,it,loss):
		if self.testLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_testloss".format(opt.model),
						  opts={ "title": "{0} (TEST_error)".format(opt.model) })
			self.testLossInit = False
		else: self.vis.line(Y=np.array([loss]),X=np.array([it]),win=opt.model+"_testloss",update="append")
	
	def validLoss(self,opt,it,loss):
		if self.validLossInit:
			self.vis.line(Y=np.array([loss]),X=np.array([it]),win="{0}_validloss".format(opt.model),
						  opts={ "title": "{0} (VALID_error)".format(opt.model) })
			self.validLossInit = False
		else: self.vis.line(Y=np.array([loss]),X=np.array([it]),win=opt.model+"_validloss",update="append")
	
	def meanVar(self,opt,mean,var):
		mean = [self.tileImages(opt,m,opt.H,opt.W,1,10) for m in mean]
		var = [self.tileImages(opt,v,opt.H,opt.W,1,10)*3 for v in var]
		self.vis.image(mean[0].clip(0,1),win="{0}_meaninit".format(opt.model), opts={ "title": "{0} (TEST_mean_init)".format(opt.model) })
		self.vis.image(mean[1].clip(0,1),win="{0}_meanwarped".format(opt.model), opts={ "title": "{0} (TEST_mean_warped)".format(opt.model) })
		self.vis.image(var[0].clip(0,1),win="{0}_varinit".format(opt.model), opts={ "title": "{0} (TEST_var_init)".format(opt.model) })
		self.vis.image(var[1].clip(0,1),win="{0}_varwarped".format(opt.model), opts={ "title": "{0} (TEST_var_warped)".format(opt.model) })

