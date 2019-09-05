"""
The main script which generates test data, runs DTW as explained in the slides and saves a json file for precision-recall
"""

from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import shutil
import json
import pickle
from dtw_own import dtw_own
import matplotlib.pyplot as plt
import copy


samp_rate = 16000
win_length, hop = 25, 10
downsample_rate = 16000 // samp_rate
templates_per_kw = 1
utter_per_kw = 1
total_nonkw_utter = 5
np.random.seed(7)
keywords = sorted(['rare', 'ambulance', 'money', 'water'])
kw_path = 'keywords/'
results_json = 'results/results_8.json'
roc_pickle_pth = 'results/roc_8.pkl'
sequence_method = 'own'


def listdir(x):
    return sorted([x for x in os.listdir(x) if x != '.DS_Store'])


def generate_clips_kwds(base_path='../datasets/TIMIT/TRAIN/'):

    if not os.path.exists(kw_path):
        os.mkdir(kw_path)
    else:
        shutil.rmtree(kw_path)
        os.mkdir(kw_path)

    non_kw_clips = []
    kw_clips = {k: [] for k in keywords}

    for dialect in listdir(base_path):
        print("Dialect:", dialect)

        for speaker in listdir(base_path + dialect):
            bp = base_path + dialect + '/' + speaker + '/'

            for file in listdir(bp):
                if file.split('.')[-1] == 'WRD':
                    with open(bp + file, 'r') as f:
                        dat = f.readlines()

                    all_dat = [x.strip().split(' ') for x in dat]
                    words = [x[-1] for x in all_dat]

                    inter = set(keywords).intersection(set(words))
                    if len(inter) == 1:
                        cur_keyword = inter.pop()
                        word_line = all_dat[words.index(cur_keyword)]
                        kw_clips[cur_keyword].append(
                            (bp + file, (int(word_line[0]) // downsample_rate, int(word_line[1]) // downsample_rate))
                        )
                    else:
                        non_kw_clips.append(bp + file)

    chosen_kw_clips = {k: [] for k in keywords}
    kw_templates = {k: [] for k in keywords}

    for word, l in kw_clips.items():
        idxs = np.random.choice(len(l), utter_per_kw + templates_per_kw)
        for idx in idxs[:utter_per_kw]:
            chosen_kw_clips[word].append(l[idx][0])
        for idx in idxs[utter_per_kw:]:
            kw_templates[word].append(l[idx])

    # Save keyword wav files
    for word, l in kw_templates.items():
        for idx in range(len(l)):
            save_name = kw_path + word + '_' + str(idx) + '.wav'
            start_t, stop_t = l[idx][1]
            wav_path = l[idx][0][:-3] + 'wav'
            (rate, sig) = wav.read(wav_path)
            only_kw = sig[start_t:stop_t]
            wav.write(save_name, rate, only_kw)

    np.random.shuffle(non_kw_clips)
    print(
        "non-kw clips: {0} ; kw-clips: {1}".format(len(non_kw_clips), sum([len(x) for x in chosen_kw_clips.values()])))
    return non_kw_clips[:total_nonkw_utter], chosen_kw_clips


# generate_clips()
# exit(0)

def dist_func(x, y, func='euclidean'):

    if func == 'euclidean':
        return np.sqrt(np.sum((x - y) ** 2))
    elif func == 'cosine':
        dot = np.dot(x, y)
        return 1 - dot / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        print("Distance func not implemented")
        exit(0)


def sample_to_frame(num, rate=samp_rate, window=win_length, hop=hop):
    multi = rate / 1000
    if num < window * multi:
        return 0
    else:
        return (num - multi * window) // (multi * hop) + 1


def sequence_from_thresh(match):
    if len(match) == 0:
        print("Couldn't find any audio in input clip")
        exit(0)

    sequences = []
    cur_seq = [match[0]]
    cur_id = 1

    while cur_id < len(match):
        if match[cur_id] == match[cur_id - 1] + 1:
            cur_seq.append(match[cur_id])
            if cur_id == len(match) - 1:
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


def proc_one(filename):
    (rate, sig) = wav.read(filename)
    assert rate == samp_rate
    # since templates have max value of 32768, normalise it
    if sig.max() > 1:
        sig = sig / 32768
    sig = sig / max(sig)
    # calculate mel filterbank energies
    feat = mfcc(sig, samplerate=samp_rate, winlen=win_length / 1000, winstep=hop / 1000, preemph=0.95, numcep=14,
                winfunc=np.hamming)
    return feat


# generates the LMD table by DTW calculations
def compare_all(clip, template):

    temp_l = template.shape[0]
    clip_l = clip.shape[0]
    print("Length of template:", temp_l)
    lower, upper = int(temp_l * 0.5), int(2 * temp_l)  # controlled by the variation in the human speech rate
    total_tries = clip_l - lower  # number of starting frames to check

    lmd = {}  # key is the starting frame while value is the minimum distance

    # if length of clip < lower_limit*(length of template), total tries = 0
    if total_tries < 0:
        total_tries = 1

    distances_matrix = np.zeros((clip_l, temp_l))
    # calculated distance matrix and feed it to DTW to avoid repeated calculations
    for i in range(clip_l):
        for j in range(temp_l):
            distances_matrix[i, j] = dist_func(clip[i], template[j])

    print("Total starting frames to check:", total_tries)

    for start_frame in range(0, total_tries):

        distances = []

        for length in range(lower, upper + 1):

            if start_frame + length > clip_l:  # Avoid repeated calculations at the end
                break

            clip_to_check = clip[start_frame:start_frame + length, :]  # consider only a slice of the total clip
            dtw_cost = dtw_own(clip_to_check, template, distances=distances_matrix[start_frame:start_frame + length, :])
            distances.append((dtw_cost, length))

        # append minimum distance to table
        lmd[start_frame] = sorted(distances, key=lambda x: x[0])[0]

        # print progress every 20 frames
        if start_frame % 20 == 0:
            print("Starting frame:", start_frame)
            print("Min distance:", lmd[start_frame])

    return lmd


# plot histogram given the LMD table, standard deviation and mode
def plot_hist(LMD, mode, std, sent=None, kw=None, is_kw=None, path=None):

    plt.hist(LMD)
    plt.axvline(x=mode, color='r', label='Mode')
    plt.axvline(x=mode - std, color='g', label='Mode - Std. Dev')
    plt.xlabel('Distance')
    plt.ylabel("Number of occurences")
    if sent is not None and kw is not None and is_kw is not None:
        if is_kw:
            title = sent + '\n' + kw + '\n' + 'PRESENT'
        else:
            title = sent + '\n' + kw + '\n' + 'ABSENT'
        plt.title(title)

    plt.legend()
    if path is not None:
        plt.savefig(path)
        plt.clf()
    else:
        plt.show()

    return std


# takes LMD distances table and parameters as input, calculates histogram and outputs
# True or False (keyword present or not)
def hist_and_final_result(lmd_dict, std_multi, cons_k, plotting_hist=False, sent=None, kw=None, is_kw=None, path=None):

    if len(lmd_dict.keys()) == 0:
        return False

    # calculate values required for histogram plotting
    lmd_dict_costs = {int(k): v[0] for k, v in lmd_dict.items()}
    lmd_dict_starts = {int(k): v[1] for k, v in lmd_dict.items()}

    lmd_list = list(lmd_dict_costs.values())
    hist_data, edges = np.histogram(lmd_list)

    std = np.std(lmd_list)
    max_id = np.argmax(hist_data)
    mode = (edges[max_id] + edges[max_id + 1]) / 2

    # uncomment this to see the histogram
    if plotting_hist:
        plot_hist(lmd_list, mode, std, sent, kw, is_kw, path)
        return

    lmd_np = np.ones((max(lmd_dict_costs.keys()) + 1)) * np.inf
    for idx, val in lmd_dict_costs.items():
        lmd_np[idx] = val
    # find starting frames where min distance is less than threshold

    if sequence_method == 'own':
        match = np.where(lmd_np <= mode - std_multi * std)[0]
        # print(mode, std)
        # print(match)
        if len(match) == 0:
            # print("No matches found below threshold")
            return False
        # get sequence from starting frame numbers
        sequences = sequence_from_thresh(match)
        sequences = [x[1] - x[0] + 1 for x in sequences]
        # print(sequences)
        # normalise by template length
        max_seq = max(sequences)

        if max_seq >= cons_k:
            return True
        else:
            return False

    else:
        cur_sum = np.sum(lmd_list[:cons_k + 1])
        if cur_sum > 0:
            return True

        for centre in range(cons_k // 2 + 1, len(lmd_list) - (cons_k + 1) // 2):
            cur_sum += (lmd_list[centre] - lmd_list[centre - cons_k - 1])
            if cur_sum > 0:
                return True

        return False


def testing():

    non_kw_clips, kw_clips = generate_clips_kwds()
    # print(non_kw, keywords)

    non_kw_sent_dict, kw_sent_dict = {}, {}
    templates_dict = {}

    for kw in listdir(kw_path):
        templates_dict[kw] = proc_one(kw_path + kw)

    for sent in non_kw_clips:
        filename = sent[:-3] + 'wav'
        non_kw_sent_dict[filename] = proc_one(filename)

    for word, paths in kw_clips.items():
        for path in paths:
            filename = path[:-3] + 'wav'
            kw_sent_dict[filename] = (proc_one(filename), word)

    final_results = {}

    # non-keyword comparisons
    for i, (non_kw_utterance, clip_feat) in enumerate(non_kw_sent_dict.items()):

        print(i, '/', len(non_kw_sent_dict))

        final_results[non_kw_utterance] = {}

        for keyword, kw_feat in templates_dict.items():
            print("Comparing keyword and non-kw sentence:", keyword, non_kw_utterance)

            lmd = compare_all(clip_feat, kw_feat)
            final_results[non_kw_utterance][keyword] = (lmd, 0)

        with open(results_json, 'w') as f:
            json.dump(final_results, f)

    # keyword comparisons
    for i, (kw_utterance, (clip_feat, word)) in enumerate(kw_sent_dict.items()):

        print(i, '/', len(kw_sent_dict))
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


def plot_roc():

    full = True
    syl_boundary = 2

    if full:
        std_multi = [0.8, 1, 1.3, 1.5, 1.7, 1.9, 2]
        cons = [1, 4, 7, 8, 10, 12, 14]

    else:
        cons = list(range(1, syl_boundary + 2))
        std_multi = [1, 1.5, 2, 2.5, 2.7, 3, 3.3, 3.6, 3.9, 4.1, 4.4, 4.7, 5]

    if os.path.exists(roc_pickle_pth):
        with open(roc_pickle_pth, 'rb') as f:
            combined_results, indi_words_res = pickle.load(f)

    else:

        if not full:
            # Read syllables here
            syl_dict = {}
            # for line in syllables:
            #     name, _, start_time, end_time, syl = line.strip().split(' ')
            #     name = name + '.wav'
            #     if name not in syl_dict.keys():
            #         syl_dict[name] = {}
            #     syl_dict[name][sample_to_frame(int(float(start_time) * 16000))] = syl

        roc_struct = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        combined_results = {}

        for s in std_multi:
            combined_results[s] = {}
            for c in cons:
                combined_results[s][c] = copy.deepcopy(roc_struct)

        indi_words_res = {k: copy.deepcopy(combined_results) for k in keywords}

        # load results from json
        with open(results_json, 'r') as f:
            res = json.load(f)

        i = 0
        for sentence, compared_kwds in res.items():
            i += 1
            print(i, '/', len(res))

            for kw, d in compared_kwds.items():

                # if d[1] == 0:
                # 	print("Comparing non-kw sentence {0} and kw {1}".format(sentence, kw))
                # else:
                # 	print("Comparing kw sentence {0} and kw {1}".format(sentence, kw))
                word = kw.split('_')[0]

                if not full:
                    only_syl = {k: v for k, v in d[0].items() if abs(int(k) - syl_boundary) in syl_dict[sentence]}

                for s in std_multi:
                    for c in cons:

                        if full:
                            result = hist_and_final_result(d[0], s, c)
                        else:
                            result = hist_and_final_result(only_syl, s, c)

                        if d[1] == 1:  # keyword present

                            if result == True:
                                combined_results[s][c]['tp'] += 1
                                indi_words_res[word][s][c]['tp'] += 1
                            else:
                                combined_results[s][c]['fn'] += 1
                                indi_words_res[word][s][c]['fn'] += 1

                        else:
                            # print("Saving non keyword")
                            if result == True:
                                combined_results[s][c]['fp'] += 1
                                indi_words_res[word][s][c]['fp'] += 1
                            else:
                                combined_results[s][c]['tn'] += 1
                                indi_words_res[word][s][c]['tn'] += 1

        # print(final)
        with open(roc_pickle_pth, 'wb') as f:
            pickle.dump((combined_results, indi_words_res), f)

    # Plot ROC
    roc_vals = {}

    for std, compared_kwds in combined_results.items():
        for c, data in compared_kwds.items():

            if c not in roc_vals.keys():
                roc_vals[c] = []

            if data['tp'] + data['fp'] == 0:
                prec = 0
            else:
                prec = data['tp'] / (data['tp'] + data['fp'])

            if data['tp'] + data['fn'] == 0:
                recall = 0
            else:
                recall = data['tp'] / (data['tp'] + data['fn'])

            roc_vals[c].append((prec, recall))

    for c, l in roc_vals.items():
        plt.plot([x[0] for x in l], [x[1] for x in l], marker='x', label='Cons = ' + str(c))

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

    # plot individual:
    for keywd, cur_final in indi_words_res.items():
        print("On keyword:", keywd)
        roc_vals = {}

        for std, compared_kwds in cur_final.items():
            for c, data in compared_kwds.items():

                if c not in roc_vals.keys():
                    roc_vals[c] = []

                if data['tp'] + data['fp'] == 0:
                    prec = 0
                else:
                    prec = data['tp'] / (data['tp'] + data['fp'])

                if data['tp'] + data['fn'] == 0:
                    recall = 0
                else:
                    recall = data['tp'] / (data['tp'] + data['fn'])

                # print(std,c,prec,recall)
                roc_vals[c].append((prec, recall))

        # print(roc_vals)
        for c, l in roc_vals.items():
            plt.plot([x[0] for x in l], [x[1] for x in l], marker='x', label='Cons = ' + str(c))

        plt.xlabel('Precision')
        plt.ylabel('Recall')
        if full:
            plt.title("ROC curve for " + keywd)
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

    if not os.path.exists(base_path + 'present'):
        os.mkdir(base_path + 'present')
    if not os.path.exists(base_path + 'absent'):
        os.mkdir(base_path + 'absent')

    i = 0
    data = {'present': {'full': [], 'syl': []}, 'absent': {'full': [], 'syl': []}}

    for sentence, vals in res.items():

        for kw, d in vals.items():

            i += 1
            if d[1] == 1:
                print("Saving keyword,", i)
                filename = base_path + 'present/' + str(i) + '_full.png'
                std = hist_and_finalresult(d[0], std_multi, 1, sentence, kw, True, filename)
                data['present']['full'].append(std)
            # filename = base_path+'present/'+str(i)+'_syl.png'
            # std, mode = hist_and_finalresult(only_syl, std_multi, 1, sentence, kw, True, filename)
            # data['present']['syl'].append(std)
            else:
                print("Saving non keyword,", i)
                filename = base_path + 'absent/' + str(i) + '_full.png'
                std = hist_and_finalresult(d[0], std_multi, 1, sentence, kw, False, filename)
                data['absent']['full'].append(std)
            # filename = base_path+'absent/'+str(i)+'_syl.png'
            # std, mode = hist_and_finalresult(only_syl, std_multi, 1, sentence, kw, False, filename)
            # data['absent']['syl'].append(std)
    print(data)


# hist_and_finalresult(lmd, 2, 0.5)

if __name__ == '__main__':
    plot_roc()
    # testing()
    # gen_hist()
