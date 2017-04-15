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
from scipy.stats import entropy, norm
from sklearn.feature_extraction.text import CountVectorizer

from event_detection import annotations, data_fetchers, plotting, preprocessing
from event_detection.event_detector import construct_doc_to_day_matrix, construct_feature_trajectories
from event_detection.postprocessing import spectral_analysis, keywords2docids_simple


def moving_average(vector, window):
    """
    Compute the moving average along the given vector using a window of the given length.
    :param vector: the vector whose moving average to compute
    :param window: length of the window to use in the computation
    :return: moving average of length len(vector) - window + 1
    """
    weights = np.ones(window) / window
    return np.convolve(vector, weights, 'valid')


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
        curve = gaussian_curve(days, mean, std)
        bottom = math.ceil(max(0, mean - std))
        top = math.floor(min(n_days, mean + std))
        curve[0:bottom] = 0.0
        curve[top:n_days] = 0.0
        return curve

    def estimate_distribution_periodic(feature_index):
        """
        Model the feature trajectory by a mixture of (stream_length / dominant_period) Gaussian distributions.
        :param feature_index: index of the feature whose trajectory to model
        :return: feature trajectory modeled by a mixture of Gaussian distributions
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
            [norm.pdf(days, mean[0], np.sqrt(cov[0])) for mean, cov in zip(g.means_, g.covariances_)]))

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
                similarities[i, j] = max(entropy(f1, f2, base=2), entropy(f2, f1, base=2))
                similarities[j, i] = similarities[i, j]

    return similarities


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
    :return: the found events as a list of numpy arrays of feature indices
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

    # Sort feature indices by DPS in descending order.
    indices = list(sorted(range(len(feature_trajectories)), key=lambda i: dps[i], reverse=True))

    t = time()
    divergences = precompute_divergences(feature_trajectories, dp)
    logging.info('Precomputed information divergences in %fs.', time() - t)

    t = time()
    overlaps = precompute_df_overlaps(bow_matrix)
    logging.info('Precomputed document overlaps in %fs.', time() - t)

    t = time()
    events = []

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

        events.append(global_indices[event])

    logging.info('Detected %d events in %fs.', len(events), time() - t)
    return events


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

    return list(filter(lambda event: len(event) > 1,
                       unsupervised_greedy_event_detection(feature_indices, bow_slice, trajectories_slice, dps_slice,
                                                           dp_slice)))


DPS_BOUNDARY = 0.05
WINDOW = 7

DATASET = 'full'  # Dataset is shared across all document fetchers.
POS_KEYWORDS = ('N', 'V', 'A', 'D')  # Allowed POS tags for keyword extraction.
NAMES_SEPARATELY = False  # Whether the documents are pairs (document_headline, document_body) for each real document.

PICKLE_PATH = '../event_detection/pickle'
ID2WORD_PATH = os.path.join(PICKLE_PATH, 'id2word_original.pickle')
BOW_MATRIX_PATH = os.path.join(PICKLE_PATH, 'term_document_original.npz')
RELATIVE_DAYS_PATH = os.path.join(PICKLE_PATH, 'relative_days_original.pickle')

EVENT_DOCIDS_ORIGINAL_PATH = os.path.join(PICKLE_PATH, 'event_docids_original.pickle')
EVENT_SUMM_DOCS_ORIGINAL_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_original.pickle')


def main():
    total_time = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    keyword_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_KEYWORDS)
    summarization_fetcher = data_fetchers.CzechSummarizationTexts(dataset=DATASET)

    if os.path.exists(ID2WORD_PATH) and os.path.exists(BOW_MATRIX_PATH) and os.path.exists(RELATIVE_DAYS_PATH):
        t = time()
        logging.info('Deserializing id2word, bag of words matrix and relative days.')

        with open(ID2WORD_PATH, mode='rb') as f:
            id2word = pickle.load(f)

        bow_matrix = data_fetchers.load_sparse_csr(BOW_MATRIX_PATH)

        with open(RELATIVE_DAYS_PATH, mode='rb') as f:
            relative_days = pickle.load(f)

        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.

        logging.info('Deserialized id2word, bag of words matrix and relative days in %fs.', time() - t)
        logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype),
                     bow_matrix.getnnz())
        logging.info('Stream length: %d', stream_length)
    else:
        t = time()
        logging.info('Creating id2word, bag of words matrix and relative days.')

        if not os.path.exists(PICKLE_PATH):
            os.makedirs(PICKLE_PATH)

        documents = preprocessing.LemmaPreprocessor(keyword_fetcher, include_names=NAMES_SEPARATELY)
        vectorizer = CountVectorizer(min_df=30, max_df=0.9, binary=True, tokenizer=lambda doc: doc,
                                     preprocessor=lambda doc: doc, stop_words=preprocessing.CZECH_STOPWORDS)

        bow_matrix = vectorizer.fit_transform(documents)
        data_fetchers.save_sparse_csr(BOW_MATRIX_PATH, bow_matrix)

        id2word = {v: k for k, v in vectorizer.vocabulary_.items()}

        with open(ID2WORD_PATH, mode='wb') as f:
            pickle.dump(id2word, f)

        relative_days = keyword_fetcher.fetch_relative_days()
        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.
        logging.info('Stream length: %d', stream_length)

        with open(RELATIVE_DAYS_PATH, mode='wb') as f:
            pickle.dump(relative_days, f)

        logging.info('Created and serialized id2word, bag of words matrix and relative days in %fs.', time() - t)
        logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype),
                     bow_matrix.getnnz())

    if NAMES_SEPARATELY:
        relative_days = np.repeat(relative_days, 2)

    trajectories = construct_feature_trajectories(bow_matrix, relative_days)
    dps, dp = spectral_analysis(trajectories)

    # Aperiodic events
    aperiodic_events = detect_events(bow_matrix, trajectories, dps, dp, aperiodic=True)
    # plotting.plot_events(trajectories, aperiodic_events, id2word, dps, dirname='./original_aperiodic')
    logging.info('Aperiodic done')

    # Periodic events
    periodic_events = detect_events(bow_matrix, trajectories, dps, dp, aperiodic=False)
    # plotting.plot_events(trajectories, periodic_events, id2word, dps, dirname='./original_periodic')
    logging.info('Periodic done')

    events = aperiodic_events + periodic_events
    dtd = construct_doc_to_day_matrix(bow_matrix.shape[0], relative_days, names_separately=NAMES_SEPARATELY)

    if NAMES_SEPARATELY:
        logging.info('Dropping headlines.')
        bow_matrix = bow_matrix[1::2, :]
        logging.info('Headlines dropped.')

    if os.path.exists(EVENT_DOCIDS_ORIGINAL_PATH):
        logging.info('Deserializing event doc IDs.')

        with open(EVENT_DOCIDS_ORIGINAL_PATH, mode='rb') as f:
            all_docids = pickle.load(f)

        logging.info('Deserialized event doc IDs.')
    else:
        logging.info('Retrieving event doc IDs.')
        t = time()

        all_docids = keywords2docids_simple(events, trajectories, dps, dtd, bow_matrix)

        with open(EVENT_DOCIDS_ORIGINAL_PATH, mode='wb') as f:
            pickle.dump(all_docids, f)

        logging.info('Retrieved and serialized event doc IDs in %fs.', time() - t)

    del bow_matrix

    if os.path.exists(EVENT_SUMM_DOCS_ORIGINAL_PATH):
        logging.info('Deserializing full documents.')

        with open(EVENT_SUMM_DOCS_ORIGINAL_PATH, mode='rb') as f:
            events_docs_repr = pickle.load(f)

        logging.info('Deserialized full documents.')
    else:
        logging.info('Retrieving full documents.')
        t = time()

        events_docs_repr = annotations.docids2documents(all_docids, summarization_fetcher)

        with open(EVENT_SUMM_DOCS_ORIGINAL_PATH, mode='wb') as f:
            pickle.dump(events_docs_repr, f)

        logging.info('Retrieved and serialized full documents in %fs.', time() - t)

    logging.info('All done in %fs.', time() - total_time)


if __name__ == '__main__':
    logger = logging.getLogger()
    handler = logging.FileHandler('./docids_documents_original_log.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    main()
