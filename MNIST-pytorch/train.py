import numpy as np
import time,os,sys
import argparse
import util

print(util.toYellow("======================================================="))
print(util.toYellow("train.py (training on MNIST)"))
print(util.toYellow("======================================================="))

import torch
import data,graph,warp,util
import options

print(util.toMagenta("setting configurations..."))
opt = options.set(training=True)

# create directories for model output
util.mkdir("models_{0}".format(opt.group))

print(util.toMagenta("building network..."))
with torch.cuda.device(0):
	# ------ build network ------
	if opt.netType=="CNN":
		geometric = graph.Identity()
		classifier = graph.FullCNN(opt)
	elif opt.netType=="STN":
		geometric = graph.STN(opt)
		classifier = graph.CNN(opt)
	elif opt.netType=="IC-STN":
		geometric = graph.ICSTN(opt)
		classifier = graph.CNN(opt)
	elif opt.netType=="C-STN":
		geometric = graph.CSTN(opt)
		classifier = graph.CNN(opt)
	elif opt.netType=="DeSTN":
		geometric = graph.DenseCSTN(opt)
		classifier = graph.CNN(opt)

	print('*****geometric*****')
	print(geometric)
	print('***** *****')
	print('*****classifier*****')
	print(classifier)
	print('***** *****')
	
	# ------ define loss ------
	loss = torch.nn.CrossEntropyLoss()
	# ------ optimizer ------
	optimList = [{ "params": geometric.parameters(), "lr": opt.lrGP },
				 { "params": classifier.parameters(), "lr": opt.lrC }]
	optim = torch.optim.SGD(optimList)

# load data
print(util.toMagenta("loading MNIST dataset..."))
trainData,validData,testData = data.loadMNIST("data/MNIST.npz")

# visdom visualizer
vis = util.Visdom(opt)

print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()

best_valid_error = float("inf")
accompany_it = 0
accompany_test_error = float("inf")


# start session
with torch.cuda.device(0):
	geometric.train()
	classifier.train()
	if opt.fromIt!=0:
		util.restoreModel(opt,geometric,classifier,opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	print(util.toMagenta("start training..."))

	# training loop
	for i in range(opt.fromIt,opt.toIt):
		lrGP = opt.lrGP*opt.lrDecay**(i//opt.lrStep)
		lrC = opt.lrC*opt.lrDecay**(i//opt.lrStep)
		# make training batch
		batch = data.makeBatch(opt,trainData)
		image = batch["image"].unsqueeze(dim=1)
		label = batch["label"]
		# generate perturbation
		pInit = data.genPerturbations(opt)
		pInitMtrx = warp.vec2mtrx(opt,pInit)
		# forward/backprop through network
		optim.zero_grad()
		imagePert = warp.transformImage(opt,image,pInitMtrx)

		#old weights
		# old_state_dict = {}
		# for key in geometric.state_dict():
		# 	old_state_dict[key] = geometric.state_dict()[key].clone()
		
		imageWarpAll = geometric(opt,image,pInit) if opt.netType=="IC-STN" or opt.netType=="C-STN" or opt.netType=="DeSTN" else geometric(opt,imagePert)
		imageWarp = imageWarpAll[-1]
		output = classifier(opt,imageWarp)
		train_loss = loss(output,label)
		train_loss.backward()
		# run one step
		optim.step()

		# Save new params
		# new_state_dict = {}
		# for key in geometric.state_dict():
		# 	new_state_dict[key] = geometric.state_dict()[key].clone()

		# Compare params
		# changed_flag = False
		# for key in old_state_dict:
		# 	if not (old_state_dict[key] == new_state_dict[key]).all():
		# 		changed_flag = True
		# 		break
		# if changed_flag:
		# 	print('weights changed')

		if (i+1)%100==0:
			print("it. {0}/{1}  lr={3}(GP),{4}(C), loss={5}, time={2}"
				.format(util.toCyan("{0}".format(i+1)),
						opt.toIt,
						util.toGreen("{0:.2f}".format(time.time()-timeStart)),
						util.toYellow("{0:.0e}".format(lrGP)),
						util.toYellow("{0:.0e}".format(lrC)),
						util.toRed("{0:.4f}".format(train_loss.data[0]))))
		if (i+1)%200==0: vis.trainLoss(opt,i+1,train_loss)
		if (i+1)%1000==0:
			# evaluate on validation set
			validAcc,validMean,validVar = data.evalValid(opt,validData,geometric,classifier)
			validError = (1-validAcc)*100
			vis.validLoss(opt,i+1,validError)
			if opt.netType=="STN" or opt.netType=="IC-STN" or opt.netType=="C-STN" or opt.netType=="DeSTN":
				vis.meanVar(opt,validMean,validVar)

			# if current valid error is best, save model and calculate the test error
			if validError < best_valid_error:
				testAcc,testMean,testVar = data.evalTest(opt,testData,geometric,classifier)
				testError = (1-testAcc)*100
				vis.testLoss(opt,i+1,testError)
			
				accompany_it = i
				accompany_test_error = testError
				best_valid_error = validError				
			# if the best valid error has not been updated for a long time, stop training
			if i - accompany_it > opt.gapIt:
				break
			'''
			# evaluate on test set
			testAcc,testMean,testVar = data.evalTest(opt,testData,geometric,classifier)
			testError = (1-testAcc)*100
			vis.testLoss(opt,i+1,testError)
			if opt.netType=="STN" or opt.netType=="IC-STN":
				vis.meanVar(opt,testMean,testVar)
			'''
		if (i+1)%10000==0:
			util.saveModel(opt,geometric,classifier,i+1)
			print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.model,i+1)))

print(util.toYellow("======= TRAINING DONE ======="))

print('best validation error:', best_valid_error)
print('at iteration: ', accompany_it)
print('test error is:', accompany_test_error)