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
from numpy import array, zeros, full, argmin, inf, ndim
from scipy.spatial.distance import cdist
from math import isinf
# distance function used for cost
def distance(x, y, func='euclidean'):
    # print(x.shape, y.shape)

    if func == 'euclidean':
        return np.sqrt(np.sum((x - y) ** 2))
    elif func == 'cosine':
        dot = np.dot(x, y)
        return 1 - dot / (np.linalg.norm(x) * np.linalg.norm(y))
    else:
        print("Distance func not implemented")
        exit(0)

def accelerated_dtw(x, y, dist, warp=1):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    :param int warp: how many shifts are computed.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x) == 1:
        x = x.reshape(-1, 1)
    if ndim(y) == 1:
        y = y.reshape(-1, 1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    D0[1:, 1:] = cdist(x, y, dist)
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                min_list += [D0[min(i + k, r), j],
                             D0[i, min(j + k, c)]]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path


def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

samp_rate = 16000
win_length, hop = 30, 15
downsample_rate = 16000 // samp_rate
templates_per_kw = 2
utter_per_kw = 8
total_nonkw_utter = 170
np.random.seed(7)
keywords = sorted(
    ["rarely", "reflect", "academic", "program", "national", "movies", "social", "all", "equipment", "fresh"])
# keywords = ['rare']

rate, sig1 = wav.read('keywords/academic_0.wav')
if sig1.max() > 1:
    sig1 = sig1 / 32768
sig1 = sig1 / max(sig1)

# calculate mel filterbank energies
feat1 = mfcc(sig1, samplerate=samp_rate, winlen=win_length / 1000, winstep=hop / 1000, preemph=0.95, numcep=14,
            winfunc=np.hamming)
rate, sig2= wav.read('keywords/academic_1.wav')
if sig2.max() > 1:
    sig2 = sig2 / 32768
sig2 = sig2 / max(sig2)
# sig2 = np.append(sig2, np.zeros((1000, 1)))

# calculate mel filterbank energies
feat2 = mfcc(sig2, samplerate=samp_rate, winlen=win_length / 1000, winstep=hop / 1000, preemph=0.95, numcep=14,
            winfunc=np.hamming)
print(feat1.shape, feat2.shape)
# print(feat1.shape, feat2.shape)
cost, path = dtw_own(feat1, feat2, return_path=True)
print(cost, path)
print(accelerated_dtw(feat1, feat2, 'euclidean'))
# plot_path(path)