import numpy as np
import matplotlib.pyplot as plt

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
epochAmount = onsets.shape[0]*onsets.shape[1]*onsets.shape[2]
dataAmount = data.shape[0]
channelAmount = data.shape[1]

#-----------------------------------------------------------------------

if (True):
	#Dimensions the data will be reduced to
	sampleAmount = 205
	trialAmount = 4
	subtrialAmount = 2
	flashAmount = 3
	epochAmount = trialAmount * subtrialAmount * flashAmount
	dataAmount = 5000
	channelAmount = 5

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
