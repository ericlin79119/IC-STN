import numpy as np
import scipy.linalg
import torch

import util

# fit (affine) warp between two sets of points 
def fit(Xsrc,Xdst):
	ptsN = len(Xsrc)
	X,Y,U,V,O,I = Xsrc[:,0],Xsrc[:,1],Xdst[:,0],Xdst[:,1],np.zeros([ptsN]),np.ones([ptsN])
	A = np.concatenate((np.stack([X,Y,I,O,O,O],axis=1),
						np.stack([O,O,O,X,Y,I],axis=1)),axis=0)
	b = np.concatenate((U,V),axis=0)
	p1,p2,p3,p4,p5,p6 = scipy.linalg.lstsq(A,b)[0].squeeze()
	pMtrx = np.array([[p1,p2,p3],[p4,p5,p6],[0,0,1]],dtype=np.float32)
	return pMtrx

# compute composition of warp parameters
def compose(opt,p,dp):
	pMtrx = vec2mtrx(opt,p)
	dpMtrx = vec2mtrx(opt,dp)
	pMtrxNew = dpMtrx.matmul(pMtrx)
	temp = pMtrxNew.clone()
	pMtrxNew /= temp[:,2:3,2:3]
	# pMtrxNew /= pMtrxNew[:,2:3,2:3]
	pNew = mtrx2vec(opt,pMtrxNew)
	return pNew

# compute inverse of warp parameters
def inverse(opt,p):
	pMtrx = vec2mtrx(opt,p)
	pInvMtrx = pMtrx.inverse()
	pInv = mtrx2vec(opt,pInvMtrx)
	return pInv

# convert warp parameters to matrix
def vec2mtrx(opt,p):
	O = util.toTorch(np.zeros([opt.batchSize],dtype=np.float32))
	I = util.toTorch(np.ones([opt.batchSize],dtype=np.float32))
	if opt.warpType=="translation":
		tx,ty = torch.unbind(p,dim=1)
		pMtrx = torch.stack([torch.stack([I,O,tx],dim=-1),
							 torch.stack([O,I,ty],dim=-1),
							 torch.stack([O,O,I],dim=-1)],dim=1)
	if opt.warpType=="similarity":
		pc,ps,tx,ty = torch.unbind(p,dim=1)
		pMtrx = torch.stack([torch.stack([I+pc,-ps,tx],dim=-1),
							 torch.stack([ps,I+pc,ty],dim=-1),
							 torch.stack([O,O,I],dim=-1)],dim=1)
	if opt.warpType=="affine":
		p1,p2,p3,p4,p5,p6 = torch.unbind(p,dim=1)
		pMtrx = torch.stack([torch.stack([I+p1,p2,p3],dim=-1),
							 torch.stack([p4,I+p5,p6],dim=-1),
							 torch.stack([O,O,I],dim=-1)],dim=1)
	if opt.warpType=="homography":
		p1,p2,p3,p4,p5,p6,p7,p8 = torch.unbind(p,dim=1)
		pMtrx = torch.stack([torch.stack([I+p1,p2,p3],dim=-1),
							 torch.stack([p4,I+p5,p6],dim=-1),
							 torch.stack([p7,p8,I],dim=-1)],dim=1)
	return pMtrx

# convert warp matrix to parameters
def mtrx2vec(opt,pMtrx):
	row0, row1, row2 = pMtrx[:,0,:], pMtrx[:,1,:], pMtrx[:,2,:]
	e00, e01, e02 = row0[:,0], row0[:,1], row0[:,2]
	e10, e11, e12 = row1[:,0], row1[:,1], row1[:,2]
	e20, e21, e22 = row2[:,0], row2[:,1], row2[:,2]
	# [row0,row1,row2] = torch.unbind(pMtrx,dim=1)
	# [e00,e01,e02] = torch.unbind(row0,dim=1)
	# [e10,e11,e12] = torch.unbind(row1,dim=1)
	# [e20,e21,e22] = torch.unbind(row2,dim=1)
	if opt.warpType=="translation": p = torch.stack([e02,e12],dim=1)
	if opt.warpType=="similarity": p = torch.stack([e00-1,e10,e02,e12],dim=1)
	if opt.warpType=="affine": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12],dim=1)
	if opt.warpType=="homography": p = torch.stack([e00-1,e01,e02,e10,e11-1,e12,e20,e21],dim=1)
	return p

# warp the image
def transformImage(opt,image,pMtrx):
	refMtrx = util.toTorch(opt.refMtrx)
	refMtrx = refMtrx.repeat(opt.batchSize,1,1)
	transMtrx = refMtrx.matmul(pMtrx)
	# warp the canonical coordinates
	X,Y = np.meshgrid(np.linspace(-1,1,opt.W),np.linspace(-1,1,opt.H))
	X,Y = X.flatten(),Y.flatten()
	XYhom = np.stack([X,Y,np.ones_like(X)],axis=1).T
	XYhom = np.tile(XYhom,[opt.batchSize,1,1]).astype(np.float32)
	XYhom = util.toTorch(XYhom)
	XYwarpHom = transMtrx.matmul(XYhom)
	XwarpHom,YwarpHom,ZwarpHom = torch.unbind(XYwarpHom,dim=1)
	Xwarp = (XwarpHom/(ZwarpHom+1e-8)).view(opt.batchSize,opt.H,opt.W)
	Ywarp = (YwarpHom/(ZwarpHom+1e-8)).view(opt.batchSize,opt.H,opt.W)
	grid = torch.stack([Xwarp,Ywarp],dim=-1)
	# sampling with bilinear interpolation
	imageWarp = torch.nn.functional.grid_sample(image,grid,mode="bilinear")
	return imageWarp
