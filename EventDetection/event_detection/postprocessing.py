import logging
import math

import gensim
import numpy as np
import sklearn.mixture as gmm
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import normalize

WINDOW = 7  # Length of the window to use in computing the moving average.


def moving_average(vector, window):
    """
    Compute the moving average along the given vector using a window of the given length.
    :param vector: the vector whose moving average to compute
    :param window: length of the window to use in the computation
    :return: moving average of length len(vector) - window + 1
    """
    weights = np.ones(window) / window
    return np.convolve(vector, weights, 'valid')


def estimate_distribution_aperiodic(event_trajectory):
    """
    Model the event trajectory by a Gaussian curve. The parameters (mean and standard deviation) are estimated
    using non-linear least squares.
    :param event_trajectory: trajectory of the event
    :return: mean and standard deviation of the model
    """

    def gaussian_curve(value, loc, scale):
        return norm.pdf(value, loc=loc, scale=scale)

    n_days = len(event_trajectory)
    ma = moving_average(event_trajectory, WINDOW)

    ma_mean = np.mean(ma)
    ma_std = np.std(ma)

    cutoff = ma_mean + ma_std
    peak_indices = np.where(event_trajectory > cutoff)

    peak_days = peak_indices[0]
    peaks = event_trajectory[peak_indices].reshape(-1)
    peaks /= np.sum(peaks)  # Normalize the trajectory so it can be interpreted as probability.

    # Initial guess for the parameters is mu ~ center of the peak period, sigma ~ quarter of the peak period length.
    popt, pcov = curve_fit(gaussian_curve, peak_days, peaks, p0=(peak_days[len(peak_days) // 2], len(peak_days) / 4),
                           bounds=(0.0, n_days))

    return popt  # Mean, Std


def estimate_distribution_periodic(event_trajectory, event_period):
    """
    Model the event trajectory by a mixture of (stream_length / dominant_period) Cauchy distributions, whose
    shape tends to represent the peaks more closely than Gaussians due to steeper peaks and fatter tails.
    Cauchy distribution parameters are the location (GMM means are used) and half width at half maximum, which
    is computed from GMM standard deviations as HWHM = sqrt(2 * ln(2)) * sigma.
    :param event_trajectory: trajectory of the event
    :param event_period: dominant period of the event
    :return: [(loc, hwhm)] for each burst in the event -- length = stream_length / dominant_period
    """
    n_days = len(event_trajectory)
    days = np.arange(n_days).reshape(-1, 1)
    ma = moving_average(event_trajectory.reshape(-1), WINDOW)

    ma_mean = np.mean(ma)
    ma_std = np.std(ma)

    cutoff = ma_mean + ma_std
    observations = np.hstack((days, event_trajectory.reshape(-1, 1)))
    observations = observations[observations[:, 1] > cutoff, :]

    # Sometimes the cutoff is too harsh and we end up with less observations than components. In that case,
    # reduce the number of components to the number of features, since not all peaks were bursty enough.
    n_components = min(math.floor(n_days / event_period), len(observations))
    g = gmm.GaussianMixture(n_components=int(n_components), covariance_type='diag', init_params='kmeans',
                            random_state=1)
    g.fit(observations)
    e_parameters = []

    # Extract parameters.
    for mean_, cov_ in zip(g.means_, g.covariances_):
        loc = mean_[0]
        hwhm = np.sqrt(2 * np.log(2)) * np.sqrt(cov_[0])

        e_parameters.append((loc, hwhm))

    return e_parameters


def create_event_trajectory(event, feature_trajectories, dps, dp):
    """
    Create a trajectory of the given event as the sum of trajectories of its features weighted by their DPS.
    Also return the dominant period of the event, which is the most common dominant period of its features,
    since not all of them have necessarily the same periodicity.
    :param event: detected event (array of its feature indices)
    :param feature_trajectories: matrix of feature trajectories as row vectors
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :return: trajectory of the given event and its dominant period
    """
    e_feature_trajectories = feature_trajectories[event]
    e_power_spectra = dps[event]
    e_dominant_period = np.bincount(dp[event].astype(int)).argmax()
    e_trajectory = (e_feature_trajectories.T @ e_power_spectra) / np.sum(e_power_spectra)

    return e_trajectory, e_dominant_period


def keywords2documents_simple(events, feature_trajectories, dps, dp, dtd_matrix, bow_matrix):
    """
    Convert the keyword representation of events to document representation. Do this in a simple manner by using all
    documents published in an event bursty period containing all its keywords. Although this punishes having too many
    distinct keywords, it may have some information value, e.g. events with an empty document set are likely garbage.
    Would work only on lemmatized texts, obviously.
    :param events: list of events which in turn are lists of their keyword indices
    :param feature_trajectories:
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :param dtd_matrix: document-to-day matrix
    :param bow_matrix: bag-of-words matrix
    :return: list of tuples (burst_start, burst_end, burst_documents) for all bursts of each event (that is, each inner
        list represents an event and contains 1 tuple for every aperiodic event and T tuples for every periodic event
        with T = stream_length / event_period
    """

    def process_burst(loc, scale, aperiodic):
        # If an event burst starts right at day 0, this would get negative.
        start = max(math.floor(loc - scale), 0)
        # If an event burst ends at stream length, this would exceed the boundary.
        end = min(math.ceil(loc + scale), n_days - 1)

        # All documents published on burst days. There is exactly one '1' in every row.
        burst_docs_all, _ = dtd_matrix[:, start:end + 1].nonzero()

        # Documents containing at least one of the event word features.
        docs_either_words = bow_matrix[:, event]

        # Documents containing all of the event word features.
        docs_words = np.where(docs_either_words.getnnz(axis=1) == len(event))[0]

        # Documents both published on burst days and containing all event word features. Do not assume unique for
        # periodic events, as some bursty periods may overlap.
        docs_both = np.intersect1d(burst_docs_all, docs_words, assume_unique=aperiodic)

        return start, end, docs_both

    n_days = feature_trajectories.shape[1]
    documents = []

    for i, event in enumerate(events):
        event_trajectory, event_period = create_event_trajectory(event, feature_trajectories, dps, dp)
        is_aperiodic = event_period == n_days

        if is_aperiodic:
            burst_loc, burst_scale = estimate_distribution_aperiodic(event_trajectory)
            burst_start, burst_end, burst_docs = process_burst(burst_loc, burst_scale, is_aperiodic)
            documents.append([(burst_start, burst_end, burst_docs)])
            logging.info('Processed aperiodic event %d consisting of %d documents.', i, len(burst_docs))
        else:
            event_parameters = estimate_distribution_periodic(event_trajectory, event_period)
            event_bursts = []
            num_docs = 0

            for burst_loc, burst_scale in sorted(event_parameters, key=lambda item: item[0]):
                burst_start, burst_end, burst_docs = process_burst(burst_loc, burst_scale, is_aperiodic)
                event_bursts.append((burst_start, burst_end, burst_docs))
                num_docs += len(burst_docs)

            documents.append(event_bursts)
            logging.info('Processed periodic event %d consisting of %d documents.', i, num_docs)

    return documents


def keywords2documents_knn(events, feature_trajectories, dps, dp, dtd_matrix, doc2vec_model, id2word, k=None):
    """
    Convert the keyword representation of events to document representation. Do this by inferring a vector of the
    event and then using it to query the documents within the event bursty period. For each event, take `k` most
    similar documents to the query vector in terms of cosine similarity.
    :param events: list of events which in turn are lists of their keyword indices
    :param feature_trajectories:
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :param dtd_matrix: document-to-day matrix
    :param doc2vec_model:
    :param id2word: mapping of word IDs to the actual words
    :param k: number of most similar documents to return for each event or `None` to return the square root of the
        number of documents within an event bursty period
    :return: list of tuples (burst_start, burst_end, burst_documents) for all bursts of each event (that is, each inner
        list represents an event and contains 1 tuple for every aperiodic event and T tuples for every periodic event
        with T = stream_length / event_period. Each document is a pair (document_id, document_cosine_similarity) so
        that further event cleaning can be performed based on the similarities.
    """

    def process_burst(loc, scale, embedding):
        # If an event burst starts right at day 0, this would get negative.
        start = max(math.floor(loc - scale), 0)
        # If an event burst ends at stream length, this would exceed the boundary.
        end = min(math.ceil(loc + scale), n_days - 1)

        # All documents published on burst days. There is exactly one '1' in every row.
        burst_docs_all, _ = dtd_matrix[:, start:end + 1].nonzero()
        document_vectors = doc2vec_model.docvecs[burst_docs_all.tolist()]

        # Normalize to unit l2 norm, as gensim similarity queries assume the vectors are already normalized.
        normalize(document_vectors, copy=False)

        if k is None:
            num_best = round(math.sqrt(len(burst_docs_all)))
        else:
            num_best = k

        index = gensim.similarities.MatrixSimilarity(document_vectors, num_best=num_best,
                                                     num_features=doc2vec_model.vector_size)
        event_documents = index[embedding]

        return start, end, event_documents

    n_days = feature_trajectories.shape[1]
    documents = []

    for i, event in enumerate(events):
        # Normalize to unit l2 norm, as gensim similarity queries assume the vectors are already normalized.
        event_vector = doc2vec_model.infer_vector([id2word[keyword] for keyword in event], steps=5)
        event_vector /= np.linalg.norm(event_vector)

        event_trajectory, event_period = create_event_trajectory(event, feature_trajectories, dps, dp)
        is_aperiodic = event_period == n_days

        if is_aperiodic:
            burst_loc, burst_scale = estimate_distribution_aperiodic(event_trajectory)
            burst_start, burst_end, burst_docs = process_burst(burst_loc, burst_scale, event_vector)
            documents.append([(burst_start, burst_end, burst_docs)])
            logging.info(
                'Processed aperiodic event %d consisting of %d documents. Most similar: %s, least similar: %s.', i,
                len(burst_docs), str(burst_docs[0]), str(burst_docs[-1]))
        else:
            event_parameters = estimate_distribution_periodic(event_trajectory, event_period)
            event_bursts = []

            num_docs = 0
            most_similar = (None, -2)
            least_similar = (None, 2)

            for burst_loc, burst_scale in sorted(event_parameters, key=lambda item: item[0]):
                burst_start, burst_end, burst_docs = process_burst(burst_loc, burst_scale, event_vector)
                event_bursts.append((burst_start, burst_end, burst_docs))

                num_docs += len(burst_docs)
                most_similar = max(most_similar, burst_docs[0], key=lambda item: item[1])
                least_similar = min(least_similar, burst_docs[-1], key=lambda item: item[1])

            documents.append(event_bursts)
            logging.info('Processed periodic event %d consisting of %d documents. Most similar: %s, least similar: %s.',
                         i, num_docs, str(most_similar), str(least_similar))

    return documents
