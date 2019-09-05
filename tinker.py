from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import os
def listdir(x):
	return sorted([x for x in os.listdir(x) if x != '.DS_Store'])
count = 0
base_path = '../datasets/TIMIT/TEST/'

for dialect in listdir(base_path):
	for speaker in listdir(base_path+dialect):
		bp = base_path+dialect+'/'+speaker+'/'
		for file in listdir(bp):
			if file.split('.')[-1] == 'WAV':
				count += 1
print(count)

