import numpy as np
import math
import matplotlib.pyplot as plt
import mne
import os
import scipy
import utils

##############################################
#           P300 Speller data                #
##############################################
datafile = "D:/Studium/BMI/bmi2020_tasks/data/p3bci_data.npz"
D = np.load(datafile, allow_pickle=True)
data = D['data']
onsets = D['onsets']
timestamps = D['timestamps']
flashseq = D['flashseq']
targets = D['targets']

sampleAmount = 205
trialAmount = onsets.shape[0]
subtrialAmount = onsets.shape[1]
flashAmount = onsets.shape[2]
epochAmount = trialAmount * subtrialAmount * flashAmount
dataAmount = data.shape[0]
channelAmount = data.shape[1]

#-----------------------------------------------------------------------
#reduction of data for testpurposes

if (False):
	#Dimensions the data will be reduced to
	sampleAmount = 205
	trialAmount = 5
	subtrialAmount = 10
	flashAmount = 12
	epochAmount = trialAmount * subtrialAmount * flashAmount
	dataAmount = 50000
	channelAmount = 10

	#Generating smaller testdata. Reduces data to the dimensions given above, cutting everything beyond those.
	#This overrides the loaded data, so only use when testing.
	data = np.delete(data, range(dataAmount,data.shape[0]), 0)
	data = np.delete(data, range(channelAmount,data.shape[1]), 1)
	onsets = np.delete(onsets, range(trialAmount,onsets.shape[0]), 0)
	onsets = np.delete(onsets, range(subtrialAmount,onsets.shape[1]), 1)
	onsets = np.delete(onsets, range(flashAmount,onsets.shape[2]), 2)
	#this might need to be edited for last 205 values, so they are never the closest to any onset
	timestamps = np.delete(timestamps, range(dataAmount,timestamps.shape[0]), 0)
	flashseq = np.delete(flashseq, range(trialAmount,flashseq.shape[0]), 0)
	flashseq = np.delete(flashseq, range(subtrialAmount,flashseq.shape[1]), 1)
	flashseq = np.delete(flashseq, range(flashAmount,flashseq.shape[2]), 2)
	targets = np.delete(targets, range(trialAmount,targets.shape[0]), 0)

#-----------------------------------------------------------------------
#task 1 segmenting data into epochs

#The output array which contains the 3600 epochs, with 205 samples each on 10 channels
epochs = np.zeros((epochAmount, sampleAmount, channelAmount))

#creating an array that will contain abs(timestamps - onset) for each onset to 
#find the timestamp that is closest to the start of a flash
timestamps_minus_onset = []
for s in range(timestamps.shape[0]):
	timestamps_minus_onset.append(timestamps[s])
	
#Iterating over all epochs to fill the epoch array
ep = -1
tid = -1
for trial in onsets:
	tid += 1
    #Outputting progress to console
	print("Progress: " + str(round(100*tid/onsets.shape[0], 2)) + "%")
	for subtrial in trial:
		for flash in subtrial:
			ep += 1
			for s in range(len(timestamps_minus_onset)):
				timestamps_minus_onset[s] = abs(timestamps[s][0] - flash)
      #finding the index of the element which is closest to 0, so the id of the 
      #data point which is closest to the onset of the flash
			idx = timestamps_minus_onset.index(min(timestamps_minus_onset))
      #copying the subarray of data from the data point that is closest to the
      #onset of the flash with length 205. This is an array with dimensions
      #205 x 10, since it contains all the channels.
			epochs[ep][0:sampleAmount] = data[idx:idx + sampleAmount]

#-----------------------------------------------------------------------
#task 2 creating a vector containing the label if the epoch is a target epoch or not

#Vector that contains a 1 if the row/column flash contained the target  and 0 if not
isTarget = np.zeros((epochAmount, 1))

#returns 1 if the given row/colum contains the target and 0 otherwise
#flash: id of the row/column that did flash
#target: id of the target
def checkTarget(flash, target):
  #column
  if (flash >= 6):
    if (target % 6 == flash-6):
      return 1
    else:
      return 0
  #row
  else:
    if (target >= 6*flash and target < 6*(flash+1)):
      return 1
    else:
      return 0

#iterating over flashseq and filling out isTarget
ep = -1
tid = -1
for trial in flashseq:
  tid += 1
  for subtrial in trial:
    for flash in subtrial:
      ep += 1
      isTarget[ep] = checkTarget(flash, targets[0][tid])

#-----------------------------------------------------------------------
#task 3 generating ERP plots

#Matrix that contains the average of the epochs for each channel.
#It is split for target and non-target epochs.
avrgPerChannel = np.zeros((channelAmount, 2, sampleAmount))

#adding up all the epochs samples with same channel and isTarget
epochId = -1
for epoch in epochs:
	epochId += 1
	sampleId = -1
	for sample in epoch:
		sampleId += 1
		channelId = -1
		for channel in sample:
			channelId += 1
			avrgPerChannel[channelId][int(isTarget[epochId][0])][sampleId] += channel
			#if (int(isTarget[epochId][0]) == 0 and channelId == 1):
				#print(channel)

#calculate total target/non-target epoch amounts
nonTargetAmount = 0
targetAmount = 0
for epoch in isTarget:
  if (epoch == 0):
    nonTargetAmount += 1
  else:
    targetAmount += 1

#divide all epoch sample summs with total target/non-target epoch amounts
for channel in avrgPerChannel:
	#non-targets
	sampleId = -1
	for sample in channel[0]:
		sampleId += 1
		channel[0][sampleId] = sample/nonTargetAmount
	#targets
	sampleId = -1
	for sample in channel[1]:
		sampleId += 1
		channel[1][sampleId] = sample/targetAmount
    
#print(avrgPerChannel[1][0])
np.save("avrgPerChannel", avrgPerChannel)

#plotting
#generate x axis values
timeOfSample = np.arange(205) * (1000/256)
channelId = -1
for channel in avrgPerChannel:
  channelId += 1
  plt.title("ERP Plot; Channel " + str(channelId + 1))
  plt.plot(timeOfSample, channel[1], label="Target")
  plt.plot(timeOfSample, channel[0], label="Non-Target")
  plt.legend()
  plt.xlim(0, 800)
  plt.xlabel("Time [ms]")
  plt.ylabel("Amplitute [µV]")
  #plt.ion()
  plt.show()

#-----------------------------------------------------------------------
#task 4 generating topoplots

#Copied from utils.py
def chan4plot():
    # !!! Please adjust this path to point to the location where your p300speller.txt and biosemi32_mne-montage.txt is stored. 
    chanpath = "D:\Studium\BMI\bmi2020_tasks\toolbox"
    # Load channel infos (names + locations) and create info object for topo plots
    with open(os.path.join(chanpath,'p300speller.txt')) as f:
        chanlabel = f.read().splitlines()
    M = mne.channels.read_montage(os.path.join(chanpath,'biosemi32_mne-montage.txt'))  
    eeginfo = mne.create_info(chanlabel,256,'eeg',M)     
    return eeginfo

avrgPerChannel0 = np.zeros((channelAmount))
channelId = -1
for channel in avrgPerChannel:
	channelId += 1
	avrgPerChannel0[channelId] = channel[0][0]

#mne.viz.plot_topomap(avrgPerChannel0, chan4plot, show=True)

#calculating r²
r2 = np.zeros((channelAmount, sampleAmount))
channelId = -1
for channel in avrgPerChannel:
	channelId += 1
	sampleId = -1
	for sample in channel[0]:
		sampleId += 1
		r2[channelId][sampleId] = (sample - channel[1][sampleId])**2

#todo get the friggin plotting to work

#-----------------------------------------------------------------------
#task 5 data preparation for cross-validation

#I mixed up folds here, so every time fold is mentioned a datapartition is ment instead
#Amount of partitions the Data is split into
foldAmount = 5

#legacy code for the downsampling per averaging
#downEpochs = utils.downsample(epochs, 10)

epochPerFold = int(epochAmount/foldAmount)
#featurePerEpoch = int(round(downEpochs.shape[1] * channelAmount, 0))
featurePerEpoch = int(round(sampleAmount * channelAmount, 0))
#ndarray containing the prepareded epoch data (dataPartitions x epochsPerParition x features(=samples*channels))
featureVectors = np.zeros((foldAmount, epochPerFold, featurePerEpoch))
#ndarray containing the epochlabels of the perpared data (dataPartitions x epochsPerParition)
featureVectorLabels = np.zeros((foldAmount, epochPerFold))

#filling featureVectors and featureVectorLabels
epochId = -1
for epoch in epochs:
	epochId += 1
	featureId = -1
	for sample in epoch:
		for channel in sample:
			featureId += 1
			foldId = int(math.floor(epochId/epochPerFold))
			eId = int(epochId % epochPerFold)
			featureVectors[foldId][eId][featureId] = channel
			featureVectorLabels[foldId][eId] = isTarget[epochId]
			
#-----------------------------------------------------------------------
#task 8 applying pca to prepared data

"""
Applying pca to the given data, reducing principal components, so that at most masLostVariance is lost
input:
data: ndarray with shape (epochs x features)
maxLostVariance: the maximal variance that can be lost in decimal
output:
pcamat: ndarray containing the principal components with shape (features x principalComponents)
"""
def pca (data, maxLostVariance):
	cov = np.cov(np.transpose(data))
	pcamat, eigval, _ = scipy.linalg.svd(cov)
	eigSum = sum(eigval)
	loss = 0
	for lostPCs in range(len(eigval)):
		loss += eigval[-(lostPCs+1)] / eigSum
		if(loss > maxLostVariance):
			pcamat = pcamat[:, : -lostPCs or None]
			break
	return pcamat
	
#-----------------------------------------------------------------------
#task 6-8 cross-validation and fda

#steps for ROC generation
steps = 100	
	
#scaling the given scores to [0-1]
def scaleScores (scores):
	scaledScores = np.copy(scores)
	mx = np.amax(scores)
	mn = np.amin(scores)
	dif = mx - mn
	
	scoreId = -1
	for score in scores:
		scoreId += 1
		scaledScores[scoreId] = (score - mn) / dif
		
	#print("scaledScores: " + str(scaledScores))
	return scaledScores

#thresholds for dynamic subtrial limitation
m_thrs = np.empty(trialAmount)
#going in reverse over all folds
for trialId in range(foldAmount-1, -1, -1):
	#data for training (epochs x features)
	trainData = np.empty((0, featurePerEpoch))
	#labels for trining data (epochs)
	trainLabels = np.empty((0))
	foldId = -1
	for fold in featureVectors:
		foldId += 1
		#appending the training data
		if(foldId != trialId):
			trainData = np.concatenate((trainData, fold))
			trainLabels = np.concatenate((trainLabels, featureVectorLabels[foldId]))
	#doing pca on the training data and then reducing both training and test data according to the result
	principalComponents = pca(trainData, 0.001)
	trainData = np.matmul(trainData, principalComponents)
	testData = np.matmul(featureVectors[trialId], principalComponents)
	#generating the training vector and bias with the training data using fda
	fda_w, fda_b = utils.fda_train(trainData, trainLabels)
	#applying fda to the testdata
	scores, labels = utils.fda_test(testData, fda_w, fda_b)
	
	#calculating threshold for dynamyc subtrial limitation
	subtrialsPerFold = int(epochPerFold / flashAmount)
	trialsPerFold = int(subtrialsPerFold / subtrialAmount)
	trialLen = subtrialAmount * flashAmount
	for tId in range(trialsPerFold):
		#true trial id
		ttId = trialsPerFold * trialId + tId
		target = np.array([math.floor(targets[0][ttId]/6), 6+targets[0][ttId]%6])
		#scores for this trial
		sc = np.empty((subtrialAmount, flashAmount))
		for subtrialId in range(subtrialAmount):
			for flashId in range(flashAmount):
				sc[subtrialId][flashId] = scores[tId*trialLen + subtrialId*flashAmount + flashId]
		TPscore, M_thr, subtrial_index = utils.calcTPscore(sc, flashseq[ttId], target)
		m_thrs[ttId] = M_thr
	
	#generating ROC
	scaledScores = scaleScores(scores)
	bias = np.linspace(0, 1, steps)
	#0 tp 1 fp
	positives = np.empty((2, steps))
	bId = -1
	for b in bias:
		bId += 1
		biasedLabels = np.empty((scores.shape[0]))
		scoreId = -1
		for score in scaledScores:
			scoreId += 1
			if(score < b):
				biasedLabels[scoreId] = 0
			else:
				biasedLabels[scoreId] = 1
		tp, fp, fn, tn = utils.calc_confusion(biasedLabels, featureVectorLabels[trialId], 1, 0)
		positives[0][bId] = tp
		positives[1][bId] = fp
	
	#plotting ROC	
	auc = np.trapz(np.flip(positives[0]), x=np.flip(positives[1]))
	plt.title("ROC for fold " + str(trialId) + " AUC = " + str(auc))
	plt.plot(positives[1], positives[0])
	plt.xlabel("False Positives")
	plt.ylabel("True Positives")
	#plt.ion()
	plt.show()

#generating weight and bias for fda with full dataset
fv = np.concatenate(featureVectors)
pc = pca(fv, 0.001)
fda_w, fda_b = utils.fda_train(np.matmul(fv, pc), np.concatenate(featureVectorLabels))
print("fda_w = " + str(fda_w) + ", fda_b = " + str(fda_b))

#averaging over the thresholds for the dynamic subtrial limitation from all trials
M_thr = 0
for thr in m_thrs:
	M_thr += thr
M_thr /= trialAmount
print("M_thr = " + str(M_thr))
