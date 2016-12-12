from __future__ import division

import collections
import re
import time

R_C_thres = 0.999

# file = codecs.open('out2.txt', 'r','utf-8') # for original utf with diacritics, does not help much
file = open('out2.txt', 'r')
texts = file.readlines()
file.close()

# taken from winLen.xls -- the numbers of documents for the individual days
# sum(ocNum) must match with len(texts)
# docNumDecJan=[7366,6591,6780,6640,6094,3542,3405,6405,6352,6586,6537,6166,3290,3335,6531,6227,6547,6873,6060,3276,3110,5582,4824,2704,2692,2878,2922,2947,5052,4750,3887,3123,4821,3291,3293,6550,6636,6599,6601,6410,3862,3759,6684,6589,6446,6498,6193,3416,3459,6626,6615,6625,6869,6544,3709,3701,6870,6586,6838,6765,6657,3882]
docNum = [3002, 5442, 5559, 3059, 3163, 6089, 6013, 6284, 5918, 5916, 3221, 3304, 6139, 5978, 6293, 6103, 5821, 3463,
          3204, 6128, 6018, 6328, 6168, 3855, 4847, 3269, 6252, 5989, 6343, 6036, 6197, 3343, 3408, 6258, 6279, 6202,
          6215, 5798, 3359, 3105, 6224, 6093, 6253, 6349, 5975, 3064, 3141, 6076, 6227, 6242, 6218, 5804, 3568, 3347,
          6333, 6121, 6330, 6105, 6325, 3208, 3639, 6385, 6519, 6415, 6335, 5821, 3426, 3443, 6195, 6296, 6378, 6174,
          6011, 3433, 3532, 6273, 6384, 6586, 6127, 6119, 3455, 3438, 6253, 6226, 6263, 6353, 6008, 3328, 2974, 6422,
          6433, 6560, 6508, 6079, 3310, 3431, 6286, 6334, 6347, 6622, 5988, 3394, 3299, 6598, 6345, 6336, 6226, 5369,
          3143, 2932, 3239, 6237, 6257, 6273, 6226, 3682, 3633, 6615, 6567, 6414, 4094, 5606, 3542, 3498, 6439, 6212,
          6532, 3746, 5586, 3480, 3573, 6698, 6478, 6588, 6460, 5848, 3501, 3500, 6223, 6117, 6078, 6221, 5811, 3580,
          3519, 6497, 6129, 6309, 6007, 5854, 3467, 3552, 6302, 6291, 6056, 6167, 5794, 3086, 3110, 6212, 6283, 6076,
          6144, 5758, 3424, 3317, 6046, 6195, 5829, 5952, 5833, 3236, 2997, 5972, 5900, 6206, 6119, 5674, 3163, 2971,
          5907, 5944, 5801, 5673, 5313, 3264, 2986, 5499, 3084, 7636, 5650, 5517, 3220, 3117, 5617, 5928, 5554, 5687,
          5503, 3015, 2939, 5960, 5690, 5750, 6016, 5453, 3265, 3062, 5786, 6213, 5902, 5906, 5292, 3097, 3029, 5688,
          5817, 5718, 5755, 5373, 2999, 2959, 5664, 5743, 5712, 5755, 5426, 3304, 3120, 5666, 5738, 5493, 5816, 5246,
          3228, 3283, 5638, 5857, 5782, 5922, 5649, 3288, 3343, 6299, 6107, 6193, 6435, 5837, 3263, 3378, 6344, 6024,
          6240, 6013, 5627, 3333, 3367, 6133, 6114, 6012, 6287, 6180, 3409, 3432, 6161, 6206, 6350, 6112, 6197, 3555,
          3442, 6275, 6370, 6770, 6689, 6552, 3376, 3489, 6772, 6748, 6659, 6754, 6493, 3982, 3719, 6749, 6517, 6597,
          6524, 6395, 3747, 3187, 6858, 6354, 6434, 3409, 8686, 3330, 3294, 5740, 3959, 6280, 6623, 6327, 3526, 3469,
          6440, 6381, 6443, 6408, 6119, 3565, 3392, 6555, 6257, 6706, 6222, 6264, 3407, 3183, 3826, 6424, 6658, 6568,
          6065, 3427, 3466, 6558, 6653, 6452, 6501, 6111, 3463, 3570, 7366, 6591, 6780, 6640, 6094, 3542, 3405, 6405,
          6352, 6586, 6537, 6166, 3290, 3335, 6531, 6227, 6547, 6873, 6060, 3276, 3110, 5582, 4824, 2704, 2692, 2878,
          2922, 2947, 5052, 4750, 3887, 3123, 4821, 3291, 3293, 6550, 6636, 6599, 6601, 6410, 3862, 3759, 6684, 6589,
          6446, 6498, 6193, 3416, 3459, 6626, 6615, 6625, 6869, 6544, 3709, 3701, 6870, 6586, 6838, 6765, 6657, 3882]
N = round(sum(docNum) / len(docNum))  # set N to normalize the window sizes
docScale = [x / N for x in docNum]

print(texts[1].decode('string_escape'))  # pouziti pro nacteni bez codecs, zobrazuje ceske znaky

# pomala varianta pres collections
# cte ve formatu: texts = ['John likes to watch movies. Mary likes too.','John also likes to watch football games.']
bagsofwords = [collections.Counter(re.findall(r'\w+', txt)) for txt in texts]
bagsofwords[0]

# get the document sums for the individual days, measue time
start = time.time()
sumbags = sum(bagsofwords[501:1000], collections.Counter())  # time consuming, 500 docs takes 76s
end = time.time()
print(end - start)

# rychlejsi varianta pres sparse matrices
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

c = CountVectorizer(min_df=30, max_df=100000)  # ignore all the words that appear in fewer than min_df docs
# 2 months ... takes less than 1min, size 328468x450642, 30740640 stored elements for min_df=1, about 1/3 of the words
# for min_df=3, for the annual data and min_df=20 shape 2090635 x 157556
bow_matrix = c.fit_transform(texts).astype(bool)
# bow_matrix = c.fit_transform(texts[0:7366]) # sparse numpy int64 matrix originates, scipy.sparse.coo_matrix
# c.vocabulary_ # get mapping between words and column indices, for complete data extremely large to print
c.vocabulary_['rebel']  # index 127576
c.vocabulary_.keys()[127576]  # list of keys but not the same order as in coo matrix
print(c.vocabulary_.keys()[c.vocabulary_.values().index(127576)])  # this works
inv_vocabulary = {v: k for k, v in c.vocabulary_.items()}  # general vocabulary transpose
inv_vocabulary[127576]  # rebel
inv_vocabulary[183107]  # rebel


def annot_list(ind):
    return [inv_vocabulary[k] for k in ind]


bow_matrix.sum(1)  # total word counts in each document, fast
word_sums = bow_matrix.sum(0)  # the counts of each word in all documents, slower but still works
bow_matrix.sum(0).shape

# get the counts for the individual days
offset = 0
sum_matrix = np.empty([len(docNum), bow_matrix.shape[1]], dtype=int)
offset_store = [0]
for i in range(len(docNum)):
    sum_matrix[i] = bow_matrix[offset:offset + docNum[i]].sum(0)
    offset += docNum[i]
    offset_store.append(offset)

# dfidf ad He et al.: Analyzing feature trajectories for event detection
# idf ... word importance, more frequent words are less important
idf = np.log(sum(docNum) / sum_matrix.sum(0))
dfidf = sum_matrix / np.tile(docNum, [sum_matrix.shape[1], 1]).T * np.tile(idf, [len(docNum), 1])

# normalize to uniform N, round to simplify e.g. future computations of binomial coeffiecients
sum_matrix_norm = (np.round(sum_matrix / np.tile(docScale, [sum_matrix.shape[1], 1]).T)).astype(int)

# the final check, the sum of window counts must be the total sum
# the last window has to be +1 to make it work
check = sum_matrix.sum(0) - word_sums
check.sum()

from scipy.special import binom
from scipy.stats import binom as binm
from scipy.stats import logistic

# turn frequencies into probs
p_o = sum_matrix / np.tile(docNum, [sum_matrix.shape[1],
                                    1]).T  # observed word probs, check that "from __future__ import division" was called
p_o_norm = np.divide(sum_matrix, N)  # observed word probs for normalized sum_matrix


# p_j = p_o.mean(0) # mean word observed prob vector, rewrite, zero probs do not count

def get_pj_slow(obs):
    pj = np.empty(obs.shape[1])
    for i in range(len(pj)):
        pj[i] = obs[:, i][obs[:, i] > 0].mean()
    return pj


def get_pj(obs):
    obs[obs == 0] = np.nan
    return np.nanmean(obs, 0)


p_j = get_pj(p_o)


# for normalized sum_matrix only, docNum replaced by N, faster but approximate probs only
# robust towards high binom coefficients, log approach taken
def get_p_g_norm_robust(N, sum_matrix, p_j):
    # precompute binom coefs up to max number in sum_matrix        
    def mylogbc(max):
        mylogbc = np.zeros(max + 1)
        for i in range(max):
            mylogbc[i + 1] = mylogbc[i] + np.log(N - i) - np.log(i + 1)
        return mylogbc

    def mylogbin(n, p):
        return preclogbin[n] + n * np.log(p) + (N - n) * np.log(1 - p)

    mylogbin = np.vectorize(mylogbin)
    preclogbin = mylogbc(np.max(sum_matrix))
    p_g = np.empty(sum_matrix.shape)
    for words in range(p_g.shape[1]):
        p_g[:, words] = mylogbin(sum_matrix[:, words], p_j[words])
    return np.exp(p_g)


# for normalized sum_matrix only, docNum replaced by N, faster but approximate probs only
# fails for word counts and thus overflow binom coefficients!!!    
def get_p_g_norm(N, sum_matrix, p_j):
    def mybin(n, p):
        return binom(N, n) * p ** n * (1 - p) ** (N - n)

    mybin = np.vectorize(mybin)
    p_g = np.empty(sum_matrix.shape)
    for words in range(p_g.shape[1]):
        p_g[:, words] = mybin(sum_matrix[:, words], p_j[words])
    return p_g


# for normalized sum_matrix only, docNum replaced by N, faster but approximate probs only
def get_r_c_thres_norm(N, sum_matrix, p_j):
    r_c_thres = np.empty(sum_matrix.shape[1])
    for words in range(sum_matrix.shape[1]):
        r_c_thres[words] = binm.ppf(R_C_thres, N, p_j[words])
    return r_c_thres


# at the moment, the most time consuming part
# can be made faster by normalizing to the same window size -- use docScale
np.mean(p_o, axis=0)  # p_o.mean(0) changes p_o, places nans there
p_g = np.empty([len(docNum), bow_matrix.shape[
    1]])  # prob that the given number of words is observed in the given day purely randomly
r_c_thres = np.empty(p_g.shape)
for days in range(len(docNum)):
    for words in range(len(p_j)):
        p_g[days, words] = binom(docNum[days], sum_matrix[days, words]) * p_j[words] ** sum_matrix[days, words] * (1 -
                                                                                                                   p_j[
                                                                                                                       words]) ** (
                                                                                                                  docNum[
                                                                                                                      days] -
                                                                                                                  sum_matrix[
                                                                                                                      days, words])
        # find the border of R_C region
        r_c_thres[days, words] = binm.ppf(R_C_thres, docNum[days], p_j[words])

p_g_norm = get_p_g_norm_robust(N, sum_matrix_norm, p_j)
r_c_thres_norm = get_r_c_thres_norm(N, sum_matrix_norm, p_j)

# construct bursty features, start with the individual ones
p_b = np.zeros(p_o.shape)
# p_b[p_o>np.tile(p_j,[p_o.shape[0],1])]=1 # the initial necessary condition, not in R_A region
# p_b[sum_matrix>r_c_thres]=1 # the sufficient condition, in R_C region
p_b[sum_matrix_norm > np.tile(r_c_thres_norm,
                              [sum_matrix_norm.shape[0], 1])] = 1  # the sufficient condition, in R_C region
# sigmoid function for R_B region
# p_b[np.logical_and(sum_matrix<r_c_thres,p_o>np.tile(p_j,[p_o.shape[0],1]))]=...
for days in range(len(docNum)):
    for words in range(len(p_j)):
        # if sum_matrix[days,words]<r_c_thres[days,words] and p_o[days,words]>p_j[words]:
        #    p_b[days,words] = logistic.cdf(sum_matrix[days,words]-(r_c_thres[days,words]+p_j[words]*docNum[days])/2)
        if sum_matrix_norm[days, words] < r_c_thres_norm[words] and p_o[days, words] > p_j[words]:
            p_b[days, words] = logistic.cdf(sum_matrix_norm[days, words] - (r_c_thres_norm[words] + p_j[words] * N) / 2)

# find the most striking bursts
# there are many features with at least one non-zero burst signal ..., get their indeces
[j for (i, j) in zip(sum(p_b), range(p_b.shape[1])) if i > 0]


# find bursty features, those whose mean burstyness is more than thres times above average -- this rather promotes high frequency features
def get_bwords_mean(p_b, thres):
    bbool = p_b.mean(0) > thres * np.mean(p_b)
    return np.where(bbool)[0]


def docOverlap(bow_matrix_bin, w1, w2):
    # tests the document overlap between twor words, i.e., in what percentage of docs the words appear together
    # the calculation is limited to a specific period under consideration
    intersec = np.logical_and(bow_matrix_bin[:, w1], bow_matrix_bin[:, w2]).sum()
    if intersec > 0:
        intersec = intersec / min(bow_matrix_bin[:, w1].sum(), bow_matrix_bin[:, w2].sum())
    return (intersec)


# an alternative definition would be to find at least one strong burst -- a consecutive series of non-zero series in its p_b vector
# it makes no sense to search for individual days, nearly every word is bursty at least in one day
# thres gives the minimum length of the series (in days here)
def get_bwords_series(p_b, thres):
    def bseries(a):
        # count the longest nonzero series in array a         
        return np.max(np.diff(np.where(a == 0))) - 1

    bbool = [False] * p_b.shape[1]
    for words in range(p_b.shape[1]):
        bbool[words] = bseries(p_b[:, words]) > thres
    return np.where(bbool)[0]


def wherebseries(a):
    # find where is the longest bseries
    ls = np.diff(np.where(a == 0))[0]
    l = np.max(ls)
    # take the first occurence only
    f = np.sum(ls[0:np.where(ls == l)[0][0]])
    return [f + 1, f + l]


def wheretseries(a, thres):
    # find where are all the above-threshold series
    ls = np.diff(np.where(a == 0))[0]
    ts = np.where(ls > thres)[0]
    w = []
    for t in ts:
        # repeat for every subseries
        f = np.sum(ls[0:t])
        w.append([f + 1, f + ls[t]])
    return w


# identify events based on the overlaps between words with a long series of above-average days
# thres is the minimum length of series in days
# sim_thres is the minimum time overlap bwetween a pair of series in %
# doc_sim_thres is the minimum overlap bwetween a pair of words in the given period in terms of the particular documkents in %
def get_series_overlaps(p_b_slice, bwords, thres, sim_thres, doc_sim_thres):
    def getOverlap(a, b):
        # finds the overlap between two ranges in terms of the percentage of the size of the smaller range
        return max(0, min(a[1], b[1]) - max(a[0], b[0])) / max(a[1] - a[0], b[1] - b[0])

    def newEvent(e, eList):
        # tests whether e is contained in the list or not
        newEvent = True
        for eL in eList:
            if len(np.intersect1d(e, eL[1])) == min(len(e), len(eL[1])):
                newEvent = False
                break
        return (newEvent)
        # find all the individual bword series (i.e., the longest consecutive range where p_b is non-zero for each feature)

    # bseries is a list of lists, for each bursty word there is a list of its continuous bursty periods longer than the threshold
    bseries = []
    for b in range(p_b_slice.shape[1]):
        bseries.append(wheretseries(p_b_slice[:, b], thres))
    # make a flat equivalent
    bseries_flat = [item for sublist in bseries for item in sublist]
    bseries_len = [len(x) for x in bseries]
    # map bseries_flat back to the bursty words    
    bseries_ind = []
    for i in range(len(bseries_len)):
        bseries_ind += [i] * bseries_len[i]
    bseries_ind = np.array(bseries_ind)
    # get an overlap similarity matrix, for each pair of periods in bseries_flat
    overmat = np.triu([1.0] * len(bseries_flat))
    for r in range(len(bseries_flat)):
        for c in range(r + 1, len(bseries_flat)):
            overmat[r, c] = getOverlap(bseries_flat[r], bseries_flat[c])
            # merge the bursty periods/series into events
    # for each event, build the list of words whose overlap is higher than sim_thres
    bevents = []
    for r in range(len(bseries_flat)):  # iterate over all the bursty perioeds and all the bursty words
        e_periods = np.where(overmat[r, :] > sim_thres)[0]
        # there is a chance to build an event for the given word and period        
        if len(e_periods) > 1:
            e_words = bseries_ind[e_periods]
            # pruning based on coocurence in the same docs, relate to the core (first) word only
            period_slice = bow_matrix[offset_store[bseries_flat[r][0]]:offset_store[bseries_flat[r][1]],
                           bwords[e_words]].todense()
            todel = []
            for w in range(1, len(e_words)):  # skip the period index
                if docOverlap(period_slice, 0, w) < doc_sim_thres:
                    todel.append(w)
            e_words = np.delete(e_words, todel)
            if (newEvent(e_words, bevents) and len(e_words) > 1):
                e = [bseries_flat[r], e_words.tolist()]
                bevents.append(e)
    return (bevents)


# get a binary word vector for an event, return p(ek)
# the unioin of burst probability series is given by their total sum
# the intersection of burst probability series is given by sum of miinimum values achieved in all the windows
def get_p_ek(p_b, E):
    return np.log(sum(p_b[:, E].min(1)) / np.sum(p_b[:, E]))


# no infinity values
def get_p_ek_nolog(p_b, E):
    return sum(p_b[:, E].min(1)) / np.sum(p_b[:, E])


# get a binary word vector for the list of bursty features and a specific bursty event, return p(d|ek)
def get_p_d_ek(D_sizes_norm, D_sizes_all, E):
    # E=np.searchsorted(bwords,ewords) # find local event indices in bwords and thus D_sezes array
    negE = np.delete(np.arange(np.alen(range(len(D_sizes_norm)))), E)  # find its complement
    return np.sum(np.log(np.concatenate([D_sizes_norm[E], 1 - D_sizes_all[negE]])))


# finds the optimal event a subset of bursty features that minimizes -log(p_ek_d), ie. maximizes the prob
# employs greedy stepwise search
def find_ek(bow_slice, p_b_slice, D_sizes, D_all):
    best_p_ek_d = float("inf")
    # find the best pair of bursty features first
    for i in range(len(D_sizes)):
        for j in range(i + 1, len(D_sizes)):
            # get sum of docs that contain any of the event words
            M_size = np.sum(bow_slice[:, [i, j]].sum(
                1) > 0)  # sums the ewords in the individual documents first, then gets the number of docs that have non-zero sums
            D_sizes_norm = D_sizes / float(M_size)
            D_sizes_all = D_sizes / float(D_all)
            # M_size=np.sum(D_sizes) # not the same as previous row, uses D-sizes, does not implement union
            p_ek_d = (-get_p_ek(p_b_slice, [i, j]) - get_p_d_ek(D_sizes_norm, D_sizes_all, [i, j]))
            if p_ek_d < best_p_ek_d:
                best_p_ek_d = p_ek_d
                ek = [i, j]
                print(ek)
    # extend the best pair
    while True:
        extendable = False  # tests that the last extension helped
        for i in np.delete(range(len(D_sizes)), ek):
            ek_cand = np.concatenate([ek, [i]])
            M_size = np.sum(bow_slice[:, ek_cand].sum(
                1) > 0)  # sums the ewords in the individual documents first, then gets the number of docs that have non-zero sums
            D_sizes_norm = D_sizes / float(M_size)
            D_sizes_all = D_sizes / float(D_all)
            p_ek_d = (-get_p_ek(p_b_slice, ek_cand) - get_p_d_ek(D_sizes_norm, D_sizes_all, ek_cand))
            if p_ek_d < best_p_ek_d:
                best_p_ek_d = p_ek_d
                ek = np.concatenate([ek, i])
                extendable = True
        if not extendable:
            break
    return ek


# finds the optimal split of bursty words into events, HB event algorithm
def find_all_ek(bow_matrix, p_b, bwords):
    # get sums of docs that contain individual bursty words    
    D_sizes = np.squeeze(
        np.asarray((bow_matrix[:, bwords] > 0).sum(0)))  # squeeze and asarray to turn [1,n] matrix into array
    ek_list = []
    while True:
        ek = find_ek(bow_matrix[:, bwords], p_b[:, bwords], D_sizes, bow_matrix.shape[0])
        ek_list.append(bwords[ek])
        bwords = np.delete(bwords, ek)
        D_sizes = np.delete(D_sizes, ek)
        if len(bwords) < 2:
            break
    return (ek_list)


# performs a greedy search through a dist matrix starting at the position where
# return all the words who match the pair of seed words above threshold
def greedyMat(p_ek_mat, where, thres):
    ek = np.intersect1d(np.where(p_ek_mat[where[0], :] > thres), np.where(p_ek_mat[where[1], :] > thres))
    return (np.union1d(where, ek))


# finds the optimal split of bursty words into events, based on calculation of pairwise p_ek and greedy search
def find_all_ek_fast(bow_matrix, p_b, bwords):
    # get an p_ek pairwise matrix, for each pair of bwords
    # p_ek_mat=np.triu([0.0]*len(bwords))
    p_ek_mat = np.zeros([len(bwords), len(bwords)])
    for r in range(len(bwords)):
        for c in range(r + 1, len(bwords)):
            p_ek_mat[r, c] = get_p_ek_nolog(p_b, [r, c])
            p_ek_mat[c, r] = p_ek_mat[r, c]

    ek_list = []
    thres = np.percentile(np.triu(p_ek_mat), 95)
    while True:
        seed_val = np.max(p_ek_mat)
        if seed_val < thres:
            break
        where = np.where(p_ek_mat == seed_val)
        where = [where[0][0], where[1][0]]
        ek = greedyMat(p_ek_mat, where, thres)

        # validate against document overlap
        # find the longest common peridd for the seed
        ek_period = wherebseries(p_b[:, where].min(1))
        seed = np.where(np.logical_or(ek == where[0], ek == where[1]))
        period_slice = bow_matrix[offset_store[ek_period[0]]:offset_store[ek_period[1]], bwords[ek]].todense()
        todel = []
        for w in range(len(ek)):  # skip the period index
            if (docOverlap(period_slice, seed[0][0], w) < 0.2) | (docOverlap(period_slice, seed[0][1], w) < 0.2):
                todel.append(w)
        ek = np.delete(ek, todel)
        if len(ek) > 1:
            ek_list.append([ek_period, ek.tolist()])
            # delete the event from the p_ek_mat in order not to repeat it
            for w1 in ek:
                for w2 in ek:
                    p_ek_mat[w1, w2] = 0
        # delete the seed anyway
        p_ek_mat[where] = 0
        p_ek_mat[where[1], where[0]] = 0

    # perform DBSCAN, does not need prior k    
    # db=DBSCAN(eps=2,metric='precomputed').fit(p_ek_mat)
    # however, results seem weird    
    # labels=db.labels_

    return (ek_list)


import matplotlib.pyplot as plt


# red dashes, blue squares and green triangles
# plt.plot(range(len(docNum)), sum_matrix[:,0], 'r--', range(len(docNum)), sum_matrix[:,183107], 'bs-', range(len(docNum)), sum_matrix[:,127576], 'g^-')
def comp_freq_plot(words, mode='f', tofile=False, num='', period=''):
    l = list()
    a = annot_list(words)
    if mode == 'f':
        toplot = sum_matrix
        title = 'Word frequency in time windows'
        ylabel = 'frequency'
    elif mode == 'p':
        toplot = p_g_norm
        title = 'Prob that the given frequency was observed randomly'
        ylabel = 'rand prob'
    else:
        toplot = p_o
        title = 'Observed prob, the burst period' + period
        ylabel = 'prob'
    for i in range(len(words)):
        temp, = plt.plot(toplot[:, words[i]], label=a[i])
        l.append(temp)
    plt.xlabel('day')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend(l, a)
    if tofile:
        fig = plt.gcf()
        fig.savefig('graphs/' + num + '_' + a[0] + '.png')
        plt.close(fig)
    else:
        plt.show()


def bevents_plots(bevents, bwords):
    # generates and saves a set of plots for all the bevents
    for e in bevents:
        comp_freq_plot(bwords[e[1]], 'o', True, str(e[1][0]), str(e[0]))


# the approach ad Fung
bwords = get_bwords_mean(p_b,
                         5)  # generate a short list of bursty words, words must have x times higher mean burstyness than the average
bevents = find_all_ek(bow_matrix, p_b, bwords)  # slow, use the next call instead
bevents = find_all_ek_fast(bow_matrix, p_b, bwords)
bevents_plots(bevents, bwords)

# period-based approach
# the minimum length of bursty event
len_thres = 12
bwords = get_bwords_series(p_b, len_thres)  # generate a short list of bursty words, a high threshold selected
bevents = get_series_overlaps(p_b[:, bwords], bwords, len_thres, 0.9, 0.1)
bevents_plots(bevents, bwords)

annot_list(bwords)

comp_freq_plot(
    bwords)  # plot frequencies in the individual windows, it is not informative as e.g. weekend windows are shorter
comp_freq_plot(bwords, 'p')  # plot observed probs
eks = find_all_ek(bow_matrix, p_b, bwords)

comp_freq_plot([4025, 15300, 37341, 73946, 73617, 85483, 87802, 95719, 111648, 118184],
               'o')  # u'async', u'configuration', u'pasting', u'push', u'replac', u'size', u'variables', u'webpag'
comp_freq_plot([14441, 31984, 86332], 'o')  # u'civilista', u'hamas', u'raketa
comp_freq_plot([70027, 76395], 'o')  # u'off, u'play
comp_freq_plot([7025, 17222, 76429, 80995, 84213, 101669, 111570],
               'o')  # u'betlem', u'darek', u'pleas', u'predplatit', u'prosinec', u'stromecek', u'vanoce

comp_freq_plot(bwords[[0, 2]], 'o')
comp_freq_plot(eks[0], 'o')

l1, = plt.plot(p_g[:, 127576], label=inv_vocabulary[127576])
l2, = plt.plot(p_g[:, 183107], label=inv_vocabulary[183107])
plt.xlabel('day')
plt.ylabel('rand prob')
plt.title('Prob that the given frequency was observed randomly')
plt.grid(True)
plt.legend([l1, l2], [inv_vocabulary[127576], inv_vocabulary[183107]])
plt.show()


# fft approach
def get_dps(t):
    # for a time series finds its dominant period and dominant power spectrum
    fft = np.fft.fft(t, n=int(len(t) / 2))
    dps = max(fft)
    dp = len(t) / (np.where(fft == dps)[0][0] + 1)
    return ([fft, dp, abs(dps)])


dps = []
for w in range(dfidf.shape[1]):
    dps.append(get_dps(dfidf[:, w]))

# turning to dense format is not possible
ba = bow_matrix[0:7366].toarray()  # for the whole matrix a memory error
ba.shape
np.sum(ba, axis=0)
np.where()
