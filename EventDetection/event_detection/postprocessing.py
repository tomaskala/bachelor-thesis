import logging
import math
from time import time

import gensim
import numpy as np
import sklearn.mixture as gmm
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.preprocessing import normalize

WINDOW = 7  # Length of the window to use when computing the moving average.


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


def keywords2docids_simple(events, feature_trajectories, dps, dp, dtd_matrix, bow_matrix):
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


def keywords2docids_wmd(doc_fetcher, events, feature_trajectories, dps, dp, dtd_matrix, w2v_model, id2word, k=None):
    """
    Convert the keyword representation of events to document representation. Do this by retrieving the documents within
    each event's bursty period(s) and then querying them using the event keywords as a query. For each event, take `k`
    most similar documents to the query in terms of Word Mover's similarity (negative of Word Mover's Distance).
    :param doc_fetcher: document fetcher to use for document streaming
    :param events: list of events which in turn are lists of their keyword indices
    :param feature_trajectories: matrix of feature trajectories
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :param dtd_matrix: document-to-day matrix
    :param w2v_model: trained Word2Vec model (or Doc2Vec model with learned word embeddings)
    :param id2word: mapping of word IDs to the actual words
    :param k: number of most similar documents to return for each event or `None` to return the square root of the
        number of documents within an event bursty period
    :return: list of tuples (burst_start, burst_end, burst_documents) for all bursts of each event (that is, each inner
        list represents an event and contains 1 tuple for every aperiodic event and T tuples for every periodic event
        with T = stream_length / event_period. Each document is a pair (document_id, document_wm_similarity) so
        that further event cleaning can be performed based on the similarities. The documents of each event are sorted
        by their similarities in descending order.
    """
    t0 = time()

    # Step 1: Assemble a list of event bursty periods and IDs of all documents within each period.
    t = time()
    logging.info('Assembling documents of all bursty periods.')

    event_docids = assemble_event_documents(events, feature_trajectories, dps, dp, dtd_matrix)

    logging.info('Documents assembled in %fs.', time() - t)

    # Step 2: Convert the document IDs to actual documents.
    t = time()
    logging.info('Converting document IDs to documents.')

    event_bursts_documents = docids2headlines(event_docids, doc_fetcher)

    logging.info('Documents converted in %fs.', time() - t)

    # Step 3: Get the documents concerning each event using WM distance.
    t = time()
    logging.info('Calculating document similarities.')

    event_bursts_out = get_relevant_documents(events, event_bursts_documents, w2v_model, id2word, k)

    logging.info('Similarities computed in %fs.', time() - t)
    logging.info('Document representation computed in %fs total.', time() - t0)

    return event_bursts_out


def get_burst_docids(dtd_matrix, burst_loc, burst_scale):
    n_days = dtd_matrix.shape[1]

    # If an event burst starts right at day 0, this would get negative.
    burst_start = max(math.floor(burst_loc - burst_scale), 0)
    # If an event burst ends at stream length, this would exceed the boundary.
    burst_end = min(math.ceil(burst_loc + burst_scale), n_days - 1)

    # All documents published on burst days. There is exactly one '1' in every row.
    burst_docs, _ = dtd_matrix[:, burst_start:burst_end + 1].nonzero()

    return burst_start, burst_end, burst_docs


def assemble_event_documents(events, feature_trajectories, dps, dp, dtd_matrix):
    n_days = feature_trajectories.shape[1]
    events_out = []

    for i, event in enumerate(events):
        event_trajectory, event_period = create_event_trajectory(event, feature_trajectories, dps, dp)
        is_aperiodic = event_period == n_days

        if is_aperiodic:
            burst_loc, burst_scale = estimate_distribution_aperiodic(event_trajectory)
            burst_start, burst_end, burst_docs = get_burst_docids(dtd_matrix, burst_loc, burst_scale)
            events_out.append([(burst_start, burst_end, burst_docs)])
        else:
            event_parameters = estimate_distribution_periodic(event_trajectory, event_period)
            event_bursts = []

            # Sort the bursts by their location from stream start to end.
            for burst_loc, burst_scale in sorted(event_parameters, key=lambda item: item[0]):
                burst_start, burst_end, burst_docs = get_burst_docids(dtd_matrix, burst_loc, burst_scale)
                event_bursts.append((burst_start, burst_end, burst_docs))

            events_out.append(event_bursts)

    return events_out


def docids2headlines(events, fetcher):
    t = time()
    logging.info('Retrieving documents for %d events.', len(events))
    docids = []

    # Collect document IDs for all events altogether and retrieve them at once, so the collection is iterated only once.
    for event in events:
        for _, _, burst_docs in event:
            docids.extend(burst_docs)

    docids2heads = load_headlines(docids, fetcher)
    events_out = []

    # Redistribute the documents back to the individual events, keeping similarities if they were retrieved previously.
    for event in events:
        event_out = []

        for burst_start, burst_end, burst_docs in event:
            headlines_out = [(doc_id, docids2heads[doc_id]) for doc_id in burst_docs]
            event_out.append((burst_start, burst_end, headlines_out))

        events_out.append(event_out)

    logging.info('Retrieved event documents in %fs.', time() - t)
    return events_out


def load_headlines(docids, fetcher):
    if len(docids) == 0:
        raise ValueError('No document IDs given.')

    old_names_only = fetcher.names_only
    fetcher.names_only = True

    docids = list(sorted(set(docids)))
    headlines = []
    doc_pos = 0

    for doc_id, document in enumerate(fetcher):
        if doc_id == docids[doc_pos]:
            headlines.append(document.name)
            doc_pos += 1

        if doc_pos == len(docids):
            break

    fetcher.names_only = old_names_only
    return dict(zip(docids, headlines))


def query_corpus_wmd(corpus, keywords, w2v_model, k):
    if k is None:
        num_best = round(math.sqrt(len(corpus)))
    else:
        num_best = k

    headlines = [doc[1] for doc in corpus]  # Corpus is a list of (doc_id, doc_headline) pairs.

    # TODO: Normalize or not?
    index = gensim.similarities.WmdSimilarity(headlines, w2v_model=w2v_model, num_best=num_best)
    event_documents = index[keywords]

    return event_documents


def get_relevant_documents(events, event_bursts, w2v_model, id2word, k):
    event_bursts_out = []

    for event_id, (event, bursts) in enumerate(zip(events, event_bursts)):
        bursts_out = []
        event_keywords = [id2word[keyword_id] for keyword_id in event]

        num_docs = 0
        most_similar_headline, top_similarity = None, -math.inf
        least_similar_headline, bot_similarity = None, math.inf
        logging.disable(logging.INFO)  # Gensim loggers are super chatty.

        for burst in bursts:
            burst_start, burst_end, burst_headlines = burst
            # Local IDs with respect to the burst.
            event_burst_docids_local = query_corpus_wmd(burst_headlines, event_keywords, w2v_model, k)

            # Global IDs with respect to the whole document collection.
            event_burst_docids = [(burst_headlines[doc_id][0], doc_sim) for doc_id, doc_sim in event_burst_docids_local]
            bursts_out.append((burst_start, burst_end, event_burst_docids))

            num_docs += len(event_burst_docids)

            if event_burst_docids_local[0][1] > top_similarity:
                top_id, top_similarity = event_burst_docids_local[0]
                most_similar_headline = burst_headlines[top_id][1]

            if event_burst_docids_local[-1][1] < bot_similarity:
                bot_id, bot_similarity = event_burst_docids_local[-1]
                least_similar_headline = burst_headlines[bot_id][1]

        event_bursts_out.append(bursts_out)

        logging.disable(logging.NOTSET)  # Re-enable logging.
        event_desc = ', '.join(event_keywords) if len(event_keywords) <= 6 else ', '.join(event_keywords[:6]) + '...'
        logging.info('Processed event %d [%s] consisting of %d documents.', event_id, event_desc, num_docs)
        logging.info('Most similar headline: "%s" (sim: %f), least similar headline: "%s" (sim: %f)',
                     ', '.join(most_similar_headline), top_similarity, ', '.join(least_similar_headline),
                     bot_similarity)

    return event_bursts_out
