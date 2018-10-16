import numpy as np
import argparse
import warp
import util
import torch

def set(training):

	# parse input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("netType",		choices=["CNN","STN","IC-STN", "C-STN", "DeSTN"],		help="type of network")
	parser.add_argument("--group",					default="0",			help="name for group")
	parser.add_argument("--model",					default="test",			help="name for model instance")
	parser.add_argument("--size",					default="28x28",		help="image resolution")
	parser.add_argument("--warpType",				default="homography",	help="type of warp function on images",
																			choices=["translation","similarity","affine","homography"])
	parser.add_argument("--warpN",		type=int,	default=4,				help="number of recurrent transformations (for IC-STN)")
	parser.add_argument("--stnN",		type=int,	default=2,				help="number of independent STN module (for c-STN)")
	parser.add_argument("--stdC",		type=float,	default=0.1,			help="initialization stddev (classification network)")
	parser.add_argument("--stdGP",		type=float,	default=0.1,			help="initialization stddev (geometric predictor)")
	parser.add_argument("--pertScale",	type=float,	default=0.25,			help="initial perturbation scale")
	parser.add_argument("--transScale",	type=float,	default=0.25,			help="initial translation scale")
	if training: # training
		parser.add_argument("--port",		type=int,	default=8097,	help="port number for visdom visualization")
		parser.add_argument("--batchSize",	type=int,	default=100,	help="batch size for SGD")
		parser.add_argument("--lrC",		type=float,	default=1e-2,	help="learning rate (classification network)")
		parser.add_argument("--lrGP",		type=float,	default=None,	help="learning rate (geometric predictor)")
		parser.add_argument("--lrDecay",	type=float,	default=1.0,	help="learning rate decay")
		parser.add_argument("--lrStep",		type=int,	default=100000,	help="learning rate decay step size")
		parser.add_argument("--fromIt",		type=int,	default=0,		help="resume training from iteration number")
		parser.add_argument("--toIt",		type=int,	default=500000,	help="run training to iteration number")
		parser.add_argument("--gapIt",		type=int,	default=100000,	help="stop training if no better resluts after this number of iterations")
	else: # evaluation
		parser.add_argument("--batchSize",	type=int,	default=1,		help="batch size for evaluation")
	opt = parser.parse_args()

	if opt.lrGP is None: opt.lrGP = 0 if opt.netType=="CNN" else \
									1e-2 if opt.netType=="STN" else \
									1e-4 if opt.netType=="IC-STN" else \
									1e-4 if opt.netType=="C-STN" else \
									1e-4 if opt.netType=="DeSTN" else None

	# --- below are automatically set ---
	assert(torch.cuda.is_available()) # support only training on GPU for now
	torch.set_default_tensor_type("torch.cuda.FloatTensor")
	opt.training = training
	opt.H,opt.W = [int(x) for x in opt.size.split("x")]
	opt.visBlockSize = int(np.floor(np.sqrt(opt.batchSize)))
	opt.warpDim = 2 if opt.warpType == "translation" else \
				  4 if opt.warpType == "similarity" else \
				  6 if opt.warpType == "affine" else \
				  8 if opt.warpType == "homography" else None
	opt.labelN = 10
	opt.canon4pts = np.array([[-1,-1],[-1,1],[1,1],[1,-1]],dtype=np.float32)
	opt.image4pts = np.array([[0,0],[0,opt.H-1],[opt.W-1,opt.H-1],[opt.W-1,0]],dtype=np.float32)
	# opt.refMtrx = warp.fit(Xsrc=opt.canon4pts,Xdst=opt.image4pts)
	# https://github.com/chenhsuanlin/inverse-compositional-STN/issues/11
	opt.refMtrx = np.eye(3).astype(np.float32)
	if opt.netType=="STN": opt.warpN = 1

	print("({0}) {1}".format(
		util.toGreen("{0}".format(opt.group)),
		util.toGreen("{0}".format(opt.model))))
	print("------------------------------------------")
	print("network type: {0}, recurrent warps: {1}".format(
		util.toYellow("{0}".format(opt.netType)),
		util.toYellow("{0}".format(opt.warpN if opt.netType=="IC-STN" else "X"))))
	print("batch size: {0}, image size: {1}x{2}".format(
		util.toYellow("{0}".format(opt.batchSize)),
		util.toYellow("{0}".format(opt.H)),
		util.toYellow("{0}".format(opt.W))))
	print("warpScale: (pert) {0} (trans) {1}".format(
		util.toYellow("{0}".format(opt.pertScale)),
		util.toYellow("{0}".format(opt.transScale))))
	if training:
		print("[geometric predictor]    stddev={0}, lr={1}".format(
			util.toYellow("{0:.0e}".format(opt.stdGP)),
			util.toYellow("{0:.0e}".format(opt.lrGP))))
		print("[classification network] stddev={0}, lr={1}".format(
			util.toYellow("{0:.0e}".format(opt.stdC)),
			util.toYellow("{0:.0e}".format(opt.lrC))))
	print("------------------------------------------")
	if training:
		print(util.toMagenta("training model ({0}) {1}...".format(opt.group,opt.model)))

	return opt
