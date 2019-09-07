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
from dtw_own import dtw_own, plot_path
import matplotlib.pyplot as plt
import copy

samp_rate = 16000
win_length, hop = 30, 15  # in milliseconds
downsample_rate = 16000 // samp_rate  # if rate != 16000, the sample-level annotations of TIMIT need to be taken care of
templates_per_kw = 2  # no of templates per keyword
utter_per_kw = 8  # no of utterances which contain each keyword
total_nonkw_utter = 170  # total utterances which do not contain any of the keyword
np.random.seed(7)  # set seed for reproducible results
keywords = sorted(
    ["rarely", "reflect", "academic", "program", "national", "movies", "social", "all", "equipment", "fresh"])
# keywords = ['rare']
sequence_method = 'own'  # change to not_own to implement threshold detection as mentioned in paper

save_dtw_plots = False  # whether to save dtw plots for comparisons where model failed
dtw_plots_pth = 'dtw/'  # directory path where dtw graphs are stored
kw_path = 'keywords/'  # where keywords are stored
results_json = 'results/results_16.json'  # name of json results
roc_pickle_pth = 'results/roc_16.pkl'  # temporary dump of ROC values

if not os.path.exists(dtw_plots_pth):
    os.mkdir(dtw_plots_pth)


def listdir(x):

    return sorted([x for x in os.listdir(x) if x != '.DS_Store'])


def save_dtw_paths(list_of_paths, clip_name, kw_name):
    """
    Saves DTW plots for incorrectly classified examples
    :param list_of_paths: each element of list contains a list of (x,y) coordinates of the DTW path
    :param clip_name: name of utterance
    :param kw_name: keyword
    :return: None. Saves images in the specified directory
    """
    if len(list_of_paths) == 0:
        return

    for i, path in enumerate(list_of_paths):
        sent_prefix = '_'.join(clip_name.split('.')[-2].split('/')[3:])
        save_name = dtw_plots_pth + sent_prefix + '_' + kw_name.split('.')[0] + '_' + str(i) + '.png'
        print(save_name)
        plot_path(path, save_name)


def generate_clips_kwds(base_path='../datasets/TIMIT/TRAIN/'):
    """
    Crop out keywords and return a list of non-keyword and keyword utterances
    :param base_path: base path of TIMIT
    :return: non_kw_clips is a list of non-keyword utterances from the TIMIT TRAIN dataset
            kw_clips is a dictionary with key=keyword and value=list of paths to utterances containing the keyword
    """
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

                    # read the word level annotations and crop out the keywords
                    all_dat = [x.strip().split(' ') for x in dat]
                    words = [x[-1] for x in all_dat]

                    # check if words in utterance and keyword have ONE and ONLY ONE keyword
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
        # choose desired number fo templates and keyword utterances
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


def dist_func(x, y, func='euclidean'):
    """
    Distance function used for DTW calculations
    :param x: first vector
    :param y: second vector
    :param func: euclidean or cosine
    :return: distance between the vectors
    """
    if func == 'euclidean':
        return np.sqrt(np.sum((x - y) ** 2))
    elif func == 'cosine':
        dot = np.dot(x, y)
        return 1 - dot / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        print("Distance func not implemented")
        exit(0)


def sample_to_frame(num, rate=samp_rate, window=win_length, slide=hop):
    """
    Map an audio sample to a frame
    :param num: sample number
    :param rate: sampling_rate
    :param window: window length in ms
    :param slide: hop in ms
    :return: frame number
    e.g. if rate=16000, window=30ms, hop=15ms, then if num=479, frame=0; if num=480, frame=1, etc.
    """
    multi = rate / 1000
    if num < window * multi:
        return 0
    else:
        return (num - multi * window) // (multi * slide) + 1


def sequence_from_thresh(match):
    """
    Convert a sequence like [1,2,3,6,8,9,10,11,13,15] into [(1,3),(6),(8,11),(13),(15)]
    :param match: list of integers
    :return: collapse sequences into starting frame and ending frame
    """
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


def plot_hist(LMD, mode, std, sent=None, kw=None, is_kw=None, path=None):
    """
    Plot histogram from LMD table generated by sliding DTW
    :param LMD: lmd dictionary obtained from the previous function
    :param std: standard deviation
    :param mode: mode of the data
    :param sent: name of utterance
    :param kw: which keyword
    :param is_kw: true if keyword IS PRESENT in utterance, false otherwise
    :param path: if one wishes to save the plot, supply appropriate path
    """
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


def proc_one(filename):
    """
    First step which returns MFCC features by reading thw wav file
    :param filename: path to wav file
    :return: MFCC features of the signal
    """
    (rate, sig) = wav.read(filename)
    assert rate == samp_rate
    # since templates have max value of 32768, normalise it
    if sig.max() > 1:
        sig = sig / 32768
    # Normalise so that max-value is 1
    sig = sig / max(sig)

    # calculate MFCC
    feat = mfcc(sig, samplerate=samp_rate, winlen=win_length / 1000, winstep=hop / 1000, preemph=0.95, numcep=14,
                winfunc=np.hamming)
    # print(sig.shape, feat.shape)
    return feat
# generates the LMD table by DTW calculations


def compare_all(clip, template):
    """
    does sliding-window DTW comaprison of utterance and template and returns a dictionary with
    key=starting frame of utterance and value=minimum LMD distance, ratio of minimum distance length and template length
    if save_dtw = True, also returns the actual path of minimum distance
    :param clip: feature vector of utterance
    :param template: feature vector of template
    :return: dictionary with minimum DTW alignment cost, DTW path and ratio of min distance utterance slice and template
    """
    temp_l = template.shape[0]
    clip_l = clip.shape[0]
    print("Length of template:", temp_l)
    lower, upper = int(temp_l * 0.5), int(2 * temp_l)  # controlled by the variation in the human speech rate
    total_tries = clip_l - lower  # number of starting frames to check

    lmd = {}  # key is the starting frame while value is the minimum distance

    # if length of clip < lower_limit*(length of template), total tries = 1. Only one DTW calculation
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

            if save_dtw_plots:
                dtw_cost, cur_path = dtw_own(clip_to_check, template,
                                             distances=distances_matrix[start_frame:start_frame + length, :],
                                             return_path=True)
                distances.append((dtw_cost, length / temp_l, cur_path))
            else:
                dtw_cost = dtw_own(clip_to_check, template,
                                   distances=distances_matrix[start_frame:start_frame + length, :])
                distances.append((dtw_cost, length / temp_l))

        # append minimum distance to table
        lmd[start_frame] = sorted(distances, key=lambda x: x[0])[0]

        # print progress every 20 frames
        if start_frame % 20 == 0:
            print("Starting frame:", start_frame)
            print("Min distance:", lmd[start_frame][:2])

    return lmd
# plot histogram given the LMD table, standard deviation and mode


def hist_and_final_result(lmd_dict, std_multi, cons_k, plotting_hist=False, sent=None, kw=None, is_kw=None, path=None):
    """

    :param lmd_dict: lmd dictionary obtained from the previous function
    :param std_multi: multiplier with standard deviation
    :param cons_k: number of consecutive frames below threshold to look for
    All the parameters below are true if one wants to simply plot the histogram and see the shape
    :param plotting_hist: true for viewing the histogram
    :param sent: name of utterance
    :param kw: which keyword
    :param is_kw: true if keyword IS PRESENT in utterance, false otherwise
    :param path: if one wishes to save the plot, supply appropriate path
    :return: True if keyword present, False otherwise. If save_dtw, also returns dtw paths for least distance frames
    """
    if len(lmd_dict.keys()) == 0:
        return False

    # calculate values required for histogram plotting
    lmd_dict_costs = {int(k): v[0] for k, v in lmd_dict.items()}
    # use below dictionary to analyse the ratio of best aligned DTW
    # will give an idea about the variation in human speech rate (whether 0.5 to 2 is worth the time)
    lmd_dict_lengths = {int(k): v[1] for k, v in lmd_dict.items()}

    if save_dtw_plots:
        lmd_dict_paths = {int(k): v[2] for k, v in lmd_dict.items()}

    # generate histogram
    lmd_list = list(lmd_dict_costs.values())
    hist_data, edges = np.histogram(lmd_list)

    std = np.std(lmd_list)
    max_id = np.argmax(hist_data)
    mode = (edges[max_id] + edges[max_id + 1]) / 2

    if plotting_hist:
        plot_hist(lmd_list, mode, std, sent, kw, is_kw, path)
        return

    lmd_np = np.ones((max(lmd_dict_costs.keys()) + 1)) * np.inf
    for idx, val in lmd_dict_costs.items():
        lmd_np[idx] = val

    if sequence_method == 'own':

        # find starting frames where min distance is less than threshold
        match = np.where(lmd_np <= mode - std_multi * std)[0]
        if len(match) == 0:
            # print("No matches found below threshold")
            return False, []

        # get sequence from starting frame numbers
        sequences = sequence_from_thresh(match)
        max_id = sorted(sequences, key=lambda x: x[1] - x[0] + 1, reverse=True)[0]

        max_seq = max_id[1] - max_id[0] + 1

        if max_seq >= cons_k:

            if save_dtw_plots:
                cur_dtw_paths = [v for k, v in lmd_dict_paths.items() if max_id[1] <= k <= max_id[0]]
                return True, cur_dtw_paths
            else:
                return True
        else:
            if save_dtw_plots:
                cur_dtw_paths = [v for k, v in lmd_dict_paths.items() if max_id[1] <= k <= max_id[0]]
                return False, cur_dtw_paths
            else:
                return False
    # method mentioned in the research paper which does not make much sense
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
    """
    generates testing data, carries out the DTW alignment and stores results in a dictionary
    key=utterance_name, value = {keyword_1:(lmd, 0/1 depending on whether keyword is present or not), keyword_2:()...}
    json is constantly dumped to avoid data loss if program crashes
    """

    # lists which contains paths of keyword and non-keyword utterances
    non_kw_clips, kw_clips = generate_clips_kwds()

    non_kw_sent_dict, kw_sent_dict = {}, {}
    templates_dict = {}

    # calculate and store MFCC features in a dictionary
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
    """
    takes lmd values and stores results in precision-recall format for plotting curves
    if save_dtw_plots, it also dumps the best path to a png
    """

    # full=False if only syllable/phone level starting frames are to be used. NOT YET IMPLEMENTED
    full = True
    syl_boundary = 2

    # declare parameters to tune
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
            # Read syllable/phone level-annotations here

        roc_struct = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
        combined_results = {}

        for s in std_multi:
            combined_results[s] = {}
            for c in cons:
                combined_results[s][c] = copy.deepcopy(roc_struct)

        # also make a separate dictionary for individual keyword results
        indi_words_res = {k: copy.deepcopy(combined_results) for k in keywords}

        # load results from json
        with open(results_json, 'r') as f:
            res = json.load(f)

        i = 0
        for sentence, compared_kwds in res.items():
            i += 1
            print(i, '/', len(res))

            for kw, d in compared_kwds.items():

                # d[0] contains LMD dictionary, d[1] = 0/1 acc. to keyword absent/present respectively

                # if d[1] == 0:
                # 	print("Comparing non-kw sentence {0} and kw {1}".format(sentence, kw))
                # else:
                # 	print("Comparing kw sentence {0} and kw {1}".format(sentence, kw))
                word = kw.split('_')[0]

                if not full:
                    only_syl = {k: v for k, v in d[0].items() if abs(int(k) - syl_boundary) in syl_dict[sentence]}

                for s in std_multi:
                    for c in cons:

                        if save_dtw_plots:
                            if full:
                                result, dtw_paths = hist_and_final_result(d[0], s, c)
                            else:
                                result, dtw_paths = hist_and_final_result(only_syl, s, c)
                        else:
                            if full:
                                result = hist_and_final_result(d[0], s, c)
                            else:
                                result = hist_and_final_result(only_syl, s, c)

                        if d[1] == 1:  # keyword present

                            if result == True:

                                combined_results[s][c]['tp'] += 1
                                indi_words_res[word][s][c]['tp'] += 1
                            else:

                                if save_dtw_plots:
                                    save_dtw_paths(dtw_paths, sentence, kw)

                                combined_results[s][c]['fn'] += 1
                                indi_words_res[word][s][c]['fn'] += 1

                        else:
                            # print("Saving non keyword")
                            if result == True:

                                if save_dtw_plots:
                                    save_dtw_paths(dtw_paths, sentence, kw)

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

    # plot individual word-level graphs:
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



if __name__ == '__main__':
    testing()
    # plot_roc()


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