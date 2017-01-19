"""
Implementation of He, Chang, Lim, 2007, Analyzing feature trajectories for event detection.
"""
import logging
import math
import os
import pickle
from time import time

import numpy as np
import sklearn.mixture as gmm
from scipy.optimize import curve_fit
from scipy.stats import cauchy, entropy, norm
from sklearn.feature_extraction.text import CountVectorizer

from event_detection import data_fetchers, plotting
from event_detection.event_detector import construct_feature_trajectories, spectral_analysis
from event_detection.preprocessing import CZECH_STOPWORDS

WINDOW = 7
DPS_BOUNDARY = 0.03


def moving_average(vector, window):
    """
    Compute the moving average along the given vector using a window of the given length.
    :param vector: the vector whose moving average to compute
    :param window: length of the window to use in the computation
    :return: moving average of length len(vector) - window + 1
    """
    weights = np.ones(window) / window
    return np.convolve(vector, weights, 'valid')


def jensen_shannon_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between the two probability distributions. Jensen-Shannon divergence is
    symmetric (in contrast to Kullback-Leibler divergence) and its square root is a proper metric.
    :param p: the true probability distribution
    :param q: the theoretical probability distribution
    :return: Jensen-Shannon divergence between the two probability distributions
    """
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)


def precompute_divergences(feature_trajectories, dominant_periods):
    """
    Compute a matrix of information divergences between all pairs of feature trajectories. The matrix is symmetric
    (since Jensen-Shannon divergence is used) so it is enough to store its upper triangular part. It is stored as
    a dictionary mapping pairs of feature indices to their divergences.

    The trajectories are first truncated to keep only the bursty parts. First, we compute a moving average of the
    trajectory using the predefined window size. Then, we define cutoff = mean(moving_average) + std(moving_average).
    We keep only the points of the trajectory greater than cutoff, which are then modeled by Normal distribution
    (aperiodic features) or a mixture of Cauchy distributions (periodic features).

    Depending on whether dominant_periods[0] > floor(len(stream) / 2), aperiodic or periodic features, respectively,
    are assumed.
    :param feature_trajectories: matrix of feature trajectories as row vectors
    :param dominant_periods: dominant periods of all features matching feature_trajectories order
    :return: matrix of feature trajectory divergences represented by a dictionary
    """

    def estimate_distribution_aperiodic(feature_trajectory):
        """
        Model the feature trajectory by a Gaussian curve. The parameters (mean and standard deviation) are estimated
        using Least Squares fit.
        :param feature_trajectory: trajectory of the feature to model
        :return: feature trajectory modeled by a Gaussian curve.
        """

        def gaussian_curve(value, loc, scale):
            return norm.pdf(value, loc=loc, scale=scale)

        days = np.arange(n_days)
        ma = moving_average(feature_trajectory, WINDOW)

        ma_mean = np.mean(ma)
        ma_std = np.std(ma)

        cutoff = ma_mean + ma_std
        peak_indices = np.where(feature_trajectory > cutoff)

        peak_days = peak_indices[0]
        peaks = feature_trajectory[peak_indices].reshape(-1)
        peaks /= np.sum(peaks)  # Normalize the trajectory so it can be interpreted as probability.

        # Initial guess for the parameters is mu ~ center of the peak period, sigma ~ quarter of the peak period length.
        popt, pcov = curve_fit(gaussian_curve, peak_days, peaks,
                               p0=(peak_days[len(peak_days) // 2], len(peak_days) / 4), bounds=(0.0, n_days))

        mean, std = popt
        return gaussian_curve(days, mean, std)

    def estimate_distribution_periodic(feature_index):
        """
        Model the feature trajectory by a mixture of (stream_length / dominant_period) Cauchy distributions, whose
        shape tends to represent the peaks more closely than Gaussians due to steeper peaks and fatter tails.
        Cauchy distribution parameters are the location (GMM means are used) and half width at half maximum, which
        is computed from GMM standard deviations as HWHM = sqrt(2 * ln(2)) * sigma.
        :param feature_index: index of the feature whose trajectory to model
        :return: feature trajectory modeled by a mixture of Cauchy distributions
        """
        days = np.arange(n_days).reshape(-1, 1)
        ma = moving_average(feature_trajectories[feature_index].reshape(-1), WINDOW)

        ma_mean = np.mean(ma)
        ma_std = np.std(ma)

        cutoff = ma_mean + ma_std
        observations = np.hstack((days, feature_trajectories[feature_index].reshape(-1, 1)))
        observations = observations[observations[:, 1] > cutoff, :]

        # Sometimes the cutoff is too harsh and we end up with less observations than components. In that case,
        # reduce the number of components to the number of features, since not all peaks were bursty enough.
        n_components = min(math.floor(n_days / dominant_periods[feature_index]),
                           len(observations))
        g = gmm.GaussianMixture(n_components=int(n_components), covariance_type='diag', init_params='kmeans',
                                random_state=1)
        g.fit(observations)

        components = np.squeeze(np.array(
            [cauchy.pdf(days, mean[0], np.sqrt(2 * np.log(2)) * np.sqrt(cov[0])) for mean, cov in
             zip(g.means_, g.covariances_)]))

        # TODO: Temporary fix.
        if len(g.weights_) == 1:
            return g.weights_[0] * components

        return g.weights_ @ components

    n_features, n_days = feature_trajectories.shape

    if dominant_periods[0] > math.floor(n_days / 2):
        distributions = np.apply_along_axis(estimate_distribution_aperiodic, axis=1, arr=feature_trajectories)
    else:
        indices = np.arange(len(feature_trajectories)).reshape(-1, 1)
        distributions = np.apply_along_axis(estimate_distribution_periodic, axis=1, arr=indices)

    similarities = np.zeros((n_features, n_features), dtype=float)

    for i, f1 in enumerate(distributions):
        for j, f2 in enumerate(distributions):
            if i < j:
                similarities[i, j] = jensen_shannon_divergence(f1, f2)
                similarities[j, i] = similarities[i, j]

    return similarities

    # return {(i, j): jensen_shannon_divergence(f1, f2) for i, f1 in enumerate(distributions) for j, f2 in
    #         enumerate(distributions) if i < j}


def precompute_df_overlaps(bow_matrix):
    """
    Compute a matrix of document overlaps between all pairs of features.
    :param bow_matrix: Bag Of Words matrix output by the CountVectorizer
    :return: matrix of feature document overlaps
    """
    feature_doc_counts = np.asarray(bow_matrix.sum(axis=0)).squeeze()
    doc_set_minima = np.minimum.outer(feature_doc_counts, feature_doc_counts)

    return (bow_matrix.T @ bow_matrix) / doc_set_minima


def set_similarity(feature_indices, divergences):
    """
    Compute the similarity of a set of features using the precomputed KL-divergences. Set similarity is defined as
    the maximum of divergences between all pairs of features from the set.
    :param feature_indices: indices of the features from the set
    :param divergences: precomputed matrix of feature trajectory divergences
    :return: similarity of the set
    """
    # return max(divergences[i, j] for i in feature_indices for j in feature_indices if i < j)
    return np.max(divergences[np.ix_(feature_indices, feature_indices)])


def set_df_overlap(feature_indices, overlaps):
    """
    Compute the document overlap of a set of features using the precomputed document overlaps. Set document overlap
    is defined as the minimum of overlaps between all pairs of features from the set.
    :param feature_indices: indices of the features from the set
    :param overlaps: precomputed matrix of feature document overlaps
    :return: document overlap of the set
    """
    return np.min(overlaps[np.ix_(feature_indices, feature_indices)])


def unsupervised_greedy_event_detection(global_indices, bow_matrix, feature_trajectories, dps, dp):
    """
    The main algorithm for event detection, as described in the paper.
    :param global_indices: array of indices of the processed features with respect to the Bag Of Words matrix
    :param bow_matrix: Bag Of Words matrix output by the CountVectorizer
    :param feature_trajectories: matrix of feature trajectories as row vectors
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :return: yield the found events as numpy arrays of feature indices
    """

    def cost_function(feature_indices):
        with np.errstate(divide='ignore'):  # Denominator == 0 means no document overlap and return infinity.
            return set_similarity(feature_indices, divergences) / (
                set_df_overlap(feature_indices, overlaps) * np.sum(dps[feature_indices]))

    def minimizing_feature(event_so_far, feature_indices):
        index = feature_indices[0]
        min_cost = cost_function(event_so_far + [feature_indices[0]])

        for f in feature_indices[1:]:
            added = event_so_far + [f]
            added_cost = cost_function(added)

            if added_cost < min_cost:
                index, min_cost = f, added_cost

        return index, min_cost

    logging.info('Examining %d features.', len(feature_trajectories))

    # Sort feature indices by DPS in ascending order.
    indices = list(sorted(range(len(feature_trajectories)), key=lambda i: dps[i]))

    t = time()
    divergences = precompute_divergences(feature_trajectories, dp)
    logging.info('Precomputed information divergences in %fs.', time() - t)

    t = time()
    overlaps = precompute_df_overlaps(bow_matrix)
    logging.info('Precomputed document overlaps in %fs.', time() - t)

    t = time()
    found_events = 0

    while len(indices) > 0:
        feature = indices.pop(0)
        event = [feature]
        event_cost = 1 / dps[feature]

        while len(indices) > 0:
            m, cost = minimizing_feature(event, indices)

            if cost < event_cost:
                event.append(m)
                indices.remove(m)
                event_cost = cost
            else:
                break

        yield global_indices[event]
        found_events += 1

    logging.info('Detected %d events in %fs.', found_events, time() - t)


def detect_events(bow_matrix, feature_trajectories, dps, dp, aperiodic):
    """
    An interface over the main event detection function to prevent duplicated code.
    :param bow_matrix: Bag Of Words matrix output by the CountVectorizer
    :param feature_trajectories: matrix of feature trajectories as row vectors
    :param dps: dominant power spectra of all features matching feature_trajectories order
    :param dp: dominant periods of all features matching feature_trajectories order
    :param aperiodic: whether to detect aperiodic or periodic events
    :return: yield the found events as numpy arrays of feature indices
    """
    _, n_days = feature_trajectories.shape

    if aperiodic:
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp > math.floor(n_days / 2)))[0]
    else:
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp <= math.floor(n_days / 2)))[0]

    if len(feature_indices) == 0:
        logging.warning('No features to detect events from.')
        return []

    bow_slice = bow_matrix[:, feature_indices]
    trajectories_slice = feature_trajectories[feature_indices, :]
    dps_slice = dps[feature_indices]
    dp_slice = dp[feature_indices]

    return unsupervised_greedy_event_detection(feature_indices, bow_slice, trajectories_slice, dps_slice, dp_slice)


def main():
    total_time = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    PICKLE_PATH = './original_method_pickle'
    ID2WORD_PATH = os.path.join(PICKLE_PATH, 'vectorizer_dec_jan_full_sw.pickle')
    BOW_MATRIX_PATH = os.path.join(PICKLE_PATH, 'term_document_dec_jan_full_sw.pickle')
    RELATIVE_DAYS_PATH = os.path.join(PICKLE_PATH, 'relative_days_dec_jan_full_sw.pickle')

    if os.path.exists(ID2WORD_PATH) and os.path.exists(BOW_MATRIX_PATH) and os.path.exists(RELATIVE_DAYS_PATH):
        with open(ID2WORD_PATH, mode='rb') as f:
            id2word = pickle.load(f)

        with open(BOW_MATRIX_PATH, mode='rb') as f:
            bow_matrix = pickle.load(f)

        with open(RELATIVE_DAYS_PATH, mode='rb') as f:
            relative_days = pickle.load(f)

        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.

        logging.info('Deserialized id2word, bag of words matrix and relative days')
        logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype),
                     bow_matrix.getnnz())
        logging.info('Stream length: %d', stream_length)
    else:
        if not os.path.exists(PICKLE_PATH):
            os.makedirs(PICKLE_PATH)

        t = time()
        documents, relative_days = data_fetchers.fetch_czech_corpus_dec_jan()
        relative_days = np.array(relative_days)

        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.
        logging.info('Read input in %fs.', time() - t)
        logging.info('Stream length: %d', stream_length)

        vectorizer = CountVectorizer(min_df=30, max_df=0.9, binary=True, stop_words=CZECH_STOPWORDS)
        bow_matrix = vectorizer.fit_transform(documents)
        id2word = {v: k for k, v in vectorizer.vocabulary_.items()}

        with open(ID2WORD_PATH, mode='wb') as f:
            pickle.dump(id2word, f)

        with open(BOW_MATRIX_PATH, mode='wb') as f:
            pickle.dump(bow_matrix, f)

        with open(RELATIVE_DAYS_PATH, mode='wb') as f:
            pickle.dump(relative_days, f)

        logging.info('Created and serialized id2word, bag of words matrix and relative days in %fs.', time() - t)
        logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype),
                     bow_matrix.getnnz())

    trajectories = construct_feature_trajectories(bow_matrix, relative_days)
    dps, dp = spectral_analysis(trajectories)

    # Aperiodic events
    aperiodic_events = detect_events(bow_matrix, trajectories, dps, dp, aperiodic=True)
    plotting.plot_events(trajectories, aperiodic_events, id2word, dps, dp, dirname='../aperiodic')
    logging.info('Aperiodic done')

    # Periodic events
    periodic_events = detect_events(bow_matrix, trajectories, dps, dp, aperiodic=False)
    plotting.plot_events(trajectories, periodic_events, id2word, dps, dp, dirname='../periodic')
    logging.info('Periodic done')

    logging.info('All done in %fs.', time() - total_time)


if __name__ == '__main__':
    main()
