"""
The main script which generates test data, runs DTW as explained in the slides and saves a json file for precision-recall
"""

from python_speech_features import mfcc, logfbank, fbank
import scipy.io.wavfile as wav
import numpy as np
import os
import shutil
import json
import pickle
from dtw_own import dtw_own
import matplotlib.pyplot as plt
import copy

def listdir(x):
	return sorted([x for x in os.listdir(x) if x != '.DS_Store'])

samp_rate = 8000
win_length, hop = 45, 15
downsample_rate = 16000//samp_rate
templates_per_kw = 2
utter_per_kw = 8
total_nonkw_utter = 170
np.random.seed(7)
keywords = sorted(['rare', 'related', 'ambulance', 'problems', 'money', 'something', 'water', 'government', 'fresh', 'birth'])
kw_path='keywords/'
results_json = 'results/results_8.json'
roc_pickle_pth = 'results/roc_8.pkl'
sequence_method = 'own'

def generate_clips(base_path='../datasets/TIMIT_down/TRAIN/'):

	non_kw_clips = []
	kw_clips = {k:[] for k in keywords}
	for dialect in listdir(base_path):
		print("Dialect:", dialect)

		for speaker in listdir(base_path+dialect):
			bp = base_path+dialect+'/'+speaker+'/'
			for file in listdir(bp):
				if file.split('.')[-1] == 'WRD':
					with open(bp+file, 'r') as f:
						dat = f.readlines()
					words = [x.strip().split(' ')[-1] for x in dat]
					found_kw = False
					inter = set(keywords).intersection(set(words))
					if len(inter) == 1:
						kw_clips[inter.pop()].append(bp+file)
					else:
						non_kw_clips.append(bp+file)
						
	# print(kw_clips)
	# exit(0)
	trimmed_kw = {}
	for word, l in kw_clips.items():
		a = np.random.choice(l, utter_per_kw)
		trimmed_kw[word] = a
	
	np.random.shuffle(non_kw_clips)
	print("non-kw clips: {0} ; kw-clips: {1}".format(len(non_kw_clips), sum([len(x) for x in trimmed_kw.items()])))
	return non_kw_clips[:total_nonkw_utter], trimmed_kw
	
# generate_clips()
def generate_keywords(base_path='../datasets/TIMIT_down/TEST/'):

	if os.path.exists(kw_path):
		shutil.rmtree(kw_path)

	if not os.path.exists(kw_path):
		os.mkdir(kw_path)

	kw_clips = {k:[] for k in keywords}
	for dialect in listdir(base_path):
		print("Dialect:", dialect)
		for speaker in listdir(base_path+dialect):
			bp = base_path+dialect+'/'+speaker+'/'
			for file in listdir(bp):
				if file.split('.')[-1] == 'WRD':
					with open(bp+file, 'r') as f:
						dat = f.readlines()
					all_dat = [x.strip().split(' ') for x in dat]
					words = [x[-1] for x in all_dat]
					
					for kw in keywords:
						for cur in all_dat:
							if cur[-1] == kw:
								kw_clips[kw].append((bp+file, (int(cur[0])//downsample_rate, int(cur[1])//downsample_rate)))						
	
	for word, l in kw_clips.items():
		i=0
		idxs = np.random.randint(len(l), size=templates_per_kw)
		for idx in idxs:
			save_name = kw_path+word+'_'+str(i)+'.wav'
			i += 1
			start_t, stop_t = l[idx][1]
			wav_path = l[idx][0][:-3]+'wav'
			(rate,sig) = wav.read(wav_path)
			only_kw = sig[start_t:stop_t]
			wav.write(save_name, rate, only_kw)

def explore_intersection(base_path="../datasets/TIMIT_down/TEST/"):

	kw_test = {}
	for dialect in listdir(base_path):
		print("Dialect:", dialect)
		for speaker in listdir(base_path+dialect):
			bp = base_path+dialect+'/'+speaker+'/'
			for file in listdir(bp):
				if file.split('.')[-1] == 'WRD':
					with open(bp+file, 'r') as f:
						dat = f.readlines()
					all_dat = [x.strip().split(' ') for x in dat]
					for f in all_dat:
						if f[-1] not in kw_test.keys():
							kw_test[f[-1]] = 0
						kw_test[f[-1]] += 1
	te_words = set([x for x in kw_test.keys() if kw_test[x]>2])
	kw_train = {}
	base_path = '../datasets/TIMIT/TRAIN/'
	for dialect in listdir(base_path):
		print("Dialect:", dialect)
		for speaker in listdir(base_path+dialect):
			bp = base_path+dialect+'/'+speaker+'/'
			for file in listdir(bp):
				if file.split('.')[-1] == 'WRD':
					with open(bp+file, 'r') as f:
						dat = f.readlines()
					all_dat = [x.strip().split(' ') for x in dat]
					for f in all_dat:
						if f[-1] not in kw_train.keys():
							kw_train[f[-1]] = 0
						kw_train[f[-1]] += 1
	tr_words = set([x for x in kw_train.keys() if kw_train[x]>2])
	print(sorted(tr_words.intersection(te_words)))
	
# explore_intersection()
# exit(0)

def sample_to_frame(num, rate=samp_rate, window=win_length, hop=hop):
	multi = rate/(1000)
	if num<window*multi:
		return 0
	else:
		return (num-multi*window)//(multi*hop)+1

def dist_func(x, y, func='euclidean'):

	if func == 'euclidean':
		return np.sqrt(np.sum((x - y) ** 2))
	elif func == 'cosine':
		dot = np.dot(x, y)
		return 1-dot/(np.linalg.norm(x)*np.linalg.norm(y))
	else:
		print("Distance func not implemented")
		exit(0)

def sequence_from_thresh(match):

	if len(match) == 0:
		print("Couldn't find any audio in input clip")
		exit(0)

	sequences = []
	cur_seq = [match[0]]
	cur_id = 1

	while cur_id<len(match):
		if match[cur_id] == match[cur_id-1]+1:
			cur_seq.append(match[cur_id])
			if cur_id == len(match)-1:
				sequences.append(cur_seq)
				break
		else:
			sequences.append(cur_seq)
			cur_seq = [match[cur_id]]

		cur_id += 1
	if len(sequences) == 0:
		return [(match[0], match[0])]

	sequences = [(x[0], x[-1]) for x in sequences]

	return sequences

def proc_one(filename, is_template):

	(rate,sig) = wav.read(filename)
	assert rate==samp_rate
	#since templates have max value of 32768, normalise it
	if sig.max() > 1:
		sig = sig/32768
	sig = sig/max(sig)
	#calculate mel filterbank energies
	return mfcc(sig, samplerate=samp_rate, winlen=win_length/1000, winstep=hop/1000, preemph=0.95, numcep=14, winfunc=np.hamming)

#generates the LMD table by DTW calculations
def compare_all(clip, template):

	temp_l = template.shape[0]
	print("Length of template:",temp_l)
	lower, upper = int(temp_l*0.5), int(2*temp_l) #controlled by the variation in the human speech rate
	LMD = {} #key is the starting frame while value is the minimum distance
	#clips is a list with each element = (feature vectors, starting frame number in the actual clip)		
	clip_l = clip.shape[0]
	#number of starting frames to check
	total_tries = clip_l-lower
	
	#if length of clip < lower_limit*(length of template), pad with silence
	if total_tries < 0:
		total_tries = 1
		print("Exiting in compare")
		return {}

	distances_matrix = np.zeros((clip_l, temp_l))
	#calculated distance matrix and feed it to DTW to avoid repeated callculations
	for i in range(clip_l):
		for j in range(temp_l):
			distances_matrix[i,j] = dist_func(clip[i], template[j])

	print("Total starting frames to check:", total_tries)
	
	for start_frame in range(0, total_tries):

		distances = []

		for length in range(lower, upper+1):

			if start_frame+length > clip_l:
				break

			clip_to_check = clip[start_frame:start_frame+length, :] #consider only a slice of the total clip
			dtw_cost = dtw_own(clip_to_check, template, distances=distances_matrix[start_frame:start_frame+length, :])
			distances.append(dtw_cost)
		#append minimum distance to table
		LMD[start_frame] = min(distances)
		#print progress evry 10 frames
		if start_frame%20 == 0:
			print("Starting frame:",start_frame)
			print("Min distance:",LMD[start_frame])

	return LMD
#plot histogram given the LMD table, standard deviation and mode
def plot_hist(LMD, mode, std, sent=None, kw=None, is_kw=None, path=None):

	plt.hist(LMD)
	plt.axvline(x=mode, color='r', label='Mode')
	plt.axvline(x=mode-std, color='g', label='Mode - Std. Dev')
	plt.xlabel('Distance')
	plt.ylabel("Number of occurences")
	if sent is not None and kw is not None and is_kw is not None:
		if is_kw:
			title = sent+'\n'+kw+'\n'+'PRESENT'
		else:
			title = sent+'\n'+kw+'\n'+'ABSENT'
		plt.title(title)
	
	plt.legend()
	if path is not None:
		plt.savefig(path)
		plt.clf()
	else:
		plt.show()

	return std

#takes LMD distances table and parameters as input, calculates histogram and outputs True or False (keyword present or not)
def hist_and_finalresult(LMD_dict, std_multi, cons_K, sent=None, kw=None, is_kw=None, path=None):

	if len(LMD_dict.keys()) == 0:
		return False

	#calculate values required for histogram plotting
	LMD_dict = {int(k):v for k,v in LMD_dict.items()}
	LMD = list(LMD_dict.values())
	hist_data, edges = np.histogram(LMD)
	std = np.std(LMD)
	max_id = np.argmax(hist_data)
	mode = (edges[max_id]+edges[max_id+1])/2
	# print((mode-min(LMD))/std)
	#uncomment this to see the histogram
	# plot_hist(LMD, mode, std, sent, kw, is_kw, path)
	# return std, (mode-min(LMD))/std
	# print(LMD_dict)
	LMD = np.ones((max(LMD_dict.keys())+1))*np.inf
	for idx, val in LMD_dict.items():
		LMD[idx] = val
	#find starting frames where min distance is less than threshold

	if sequence_method == 'own':
		match = np.where(LMD <= mode - std_multi*std)[0]
		# print(mode, std)
		# print(match)
		if len(match) == 0:
			# print("No matches found below threshold")
			return False
		#get sequence from starting frame numbers
		sequences = sequence_from_thresh(match)
		sequences = [x[1]-x[0]+1 for x in sequences]
		# print(sequences)
		#normalise by templaet length
		max_seq = max(sequences)
		
		if max_seq >= cons_K:
			return True
		else:
			return False

	else:
		cur_sum = np.sum(LMD[:cons_K+1])
		if cur_sum > 0:
			return True

		for centre in range(cons_K//2+1, len(LMD)-(cons_K+1)//2):
			cur_sum += (LMD[centre]-LMD[centre-cons_K-1])
			if cur_sum > 0:
				return True

		return False

def testing():

	non_kw, keywords = generate_clips()
	# print(non_kw, keywords)
	generate_keywords()

	non_kw_sent_dict, kw_sent_dict= {}, {}
	templates_dict = {}
	
	for kw in listdir(kw_path):
		templates_dict[kw] = proc_one(kw_path+kw, True)

	for sent in non_kw:
		filename = sent[:-3]+'wav'
		non_kw_sent_dict[filename] = proc_one(filename, False)

	for word, paths in keywords.items():
		for path in paths:
			filename = path[:-3]+'wav'
			kw_sent_dict[filename] = (proc_one(filename, False), word)

	final_results = {}

	#non-keyword comparisons
	for i, (non_kw_utterance, clip_feat) in enumerate(non_kw_sent_dict.items()):
		
		print(i,'/',len(non_kw_sent_dict))

		final_results[non_kw_utterance] = {}

		for keyword, kw_feat in templates_dict.items():

			print("Comparing keyword and non-kw sentence:", keyword, non_kw_utterance)

			lmd = compare_all(clip_feat, kw_feat)
			final_results[non_kw_utterance][keyword] = (lmd,0)

		with open(results_json, 'w') as f:
			json.dump(final_results, f)

	#keyword comparisons
	for i, (kw_utterance, (clip_feat, word)) in enumerate(kw_sent_dict.items()):
		
		print(i,'/',len(kw_sent_dict))
		final_results[kw_utterance] = {}

		for keyword, kw_feat in templates_dict.items():

			print("Comparing keyword and kw sentence:", keyword, kw_utterance)

			lmd = compare_all(clip_feat, kw_feat)
			
			if keyword.split('_')[0] == word:
				final_results[kw_utterance][keyword] = (lmd, 1)
			else:
				final_results[kw_utterance][keyword] = (lmd, 0)

		with open(results_json, 'w') as f:
			json.dump(final_results, f)

def roc():

	full = True
	syl_boundary = 2

	if full:
		std_multi = [0.8, 1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.2, 2.5]
		cons = [1,4,7,8,10,12,14]
		
	else:
		cons = list(range(1,syl_boundary+2))
		std_multi = [1,1.5,2,2.5,2.7,3,3.3,3.6,3.9,4.1,4.4,4.7,5]

	if os.path.exists(roc_pickle_pth):
		with open(roc_pickle_pth, 'rb') as f:
			final, indi_words_res = pickle.load(f)

	else:

		if not full:
			with open('KaldiCSglobal_GTsyllables.ctm', 'r') as f:
				syllables = f.readlines()

			syl_dict = {}
			for line in syllables:
				name, _, start_time, end_time, syl = line.strip().split(' ')
				name = name+'.wav'
				if name not in syl_dict.keys():
					syl_dict[name] = {}
				syl_dict[name][sample_to_frame(int(float(start_time)*16000))] = syl
		
		data = {'tp':0,'fp':0,'tn':0,'fn':0}
		final = {}

		for s in std_multi:
			final[s] = {}
			for c in cons:
				final[s][c] = copy.deepcopy(data)

		indi_words_res = {k:copy.deepcopy(final) for k in keywords}
		# print(indi_words_res)
		with open(results_json, 'r') as f:
			res = json.load(f)

		i = 0
		for sentence, vals in res.items():
			i += 1
			print(i,'/',len(res))
			
			for kw, d in vals.items():

				# if d[1] == 0:
				# 	print("Comparing non-kw sentence {0} and kw {1}".format(sentence, kw))
				# else:
				# 	print("Comparing kw sentence {0} and kw {1}".format(sentence, kw))
				word = kw.split('_')[0]

				if not full:
					only_syl = {k:v for k,v in d[0].items() if abs(int(k)-syl_boundary) in syl_dict[sentence]}
				
				for s in std_multi:
					for c in cons:

						if d[1] == 1: #keyword				

							if full:
								result = hist_and_finalresult(d[0], s, c)
								if result == True:
									final[s][c]['tp'] += 1
									indi_words_res[word][s][c]['tp'] += 1
								else:
									final[s][c]['fn'] += 1
									indi_words_res[word][s][c]['fn'] += 1
							else:
								result = hist_and_finalresult(only_syl, s, c)
								if result == True:
									final[s][c]['tp'] += 1
									indi_words_res[word][s][c]['tp'] += 1
								else:
									final[s][c]['fn'] += 1
									indi_words_res[word][s][c]['fn'] += 1
						else:
							# print("Saving non keyword")

							if full:
								result = hist_and_finalresult(d[0], s, c)
								if result == True:
									final[s][c]['fp'] += 1
									indi_words_res[word][s][c]['fp'] += 1
								else:
									final[s][c]['tn'] += 1
									indi_words_res[word][s][c]['tn'] += 1

							else:
								result = hist_and_finalresult(only_syl, s, c)
								if result == True:
									final[s][c]['fp'] += 1
									indi_words_res[word][s][c]['fp'] += 1
								else:
									final[s][c]['tn'] += 1
									indi_words_res[word][s][c]['tn'] += 1

		# print(final)
		with open(roc_pickle_pth, 'wb') as f:
			pickle.dump((final, indi_words_res), f)
	#Plot ROC		
	roc_vals = {}
	print(indi_words_res)
	for std, vals in final.items():
		for c, data in vals.items():
			if c not in roc_vals.keys():
				roc_vals[c] = []
			
			if data['tp']+data['fp'] == 0:
				prec = 0
			else:
				prec = data['tp']/(data['tp']+data['fp'])

			if data['tp'] + data['fn'] == 0:
				recall = 0
			else:
				recall = data['tp']/(data['tp']+data['fn'])

			# print(std,c,prec,recall)
			roc_vals[c].append((prec,recall))

	# print(roc_vals)
	for c,l in roc_vals.items():
		plt.plot([x[0] for x in l], [x[1] for x in l], marker='x', label='Cons = '+str(c))
	
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	if full:
		plt.title("ROC curve for ALL")
	else:
		plt.title("ROC curve for SYLLABLES")
	plt.legend()
	plt.grid(True)
	plt.show()
	plt.clf()
	#plot individual:
	for keywd, cur_final in indi_words_res.items():
		print("On keywd:",keywd)
		roc_vals = {}
		for std, vals in cur_final.items():
			for c, data in vals.items():
				if c not in roc_vals.keys():
					roc_vals[c] = []
				
				if data['tp']+data['fp'] == 0:
					prec = 0
				else:
					prec = data['tp']/(data['tp']+data['fp'])

				if data['tp'] + data['fn'] == 0:
					recall = 0
				else:
					recall = data['tp']/(data['tp']+data['fn'])

				# print(std,c,prec,recall)
				roc_vals[c].append((prec,recall))

		# print(roc_vals)
		for c,l in roc_vals.items():
			plt.plot([x[0] for x in l], [x[1] for x in l], marker='x', label='Cons = '+str(c))
		
		plt.xlabel('Precision')
		plt.ylabel('Recall')
		if full:
			plt.title("ROC curve for "+keywd)
		else:
			plt.title("ROC curve for SYLLABLES")
		plt.legend()
		plt.grid(True)
		plt.show()
		plt.clf()

def gen_hist(base_path='histograms/'):

	std_multi = 1

	with open(results_json, 'r') as f:
		res = json.load(f)

	if not os.path.exists(base_path):
		os.mkdir(base_path)

	if not os.path.exists(base_path+'present'):
		os.mkdir(base_path+'present')
	if not os.path.exists(base_path+'absent'):
		os.mkdir(base_path+'absent')

	i = 0
	data = {'present':{'full':[],'syl':[]},'absent':{'full':[],'syl':[]}}

	for sentence, vals in res.items():

		for kw, d in vals.items():

			i += 1
			if d[1] == 1:
				print("Saving keyword,",i)
				filename = base_path+'present/'+str(i)+'_full.png'
				std = hist_and_finalresult(d[0], std_multi, 1, sentence, kw, True, filename)
				data['present']['full'].append(std)
				# filename = base_path+'present/'+str(i)+'_syl.png'
				# std, mode = hist_and_finalresult(only_syl, std_multi, 1, sentence, kw, True, filename)
				# data['present']['syl'].append(std)
			else:
				print("Saving non keyword,",i)
				filename = base_path+'absent/'+str(i)+'_full.png'
				std = hist_and_finalresult(d[0], std_multi, 1, sentence, kw, False, filename)
				data['absent']['full'].append(std)
				# filename = base_path+'absent/'+str(i)+'_syl.png'
				# std, mode = hist_and_finalresult(only_syl, std_multi, 1, sentence, kw, False, filename)
				# data['absent']['syl'].append(std)
	print(data)

	# hist_and_finalresult(lmd, 2, 0.5)

if __name__ == '__main__':
	roc()
	# testing()
	# gen_hist()
