"""
Implementation of He, Chang, Lim, 2007, Analyzing feature trajectories for event detection.
"""
import logging
import math
from time import time

import numpy as np
import scipy.sparse as sp
import sklearn.mixture as gmm
from scipy.optimize import curve_fit
from scipy.signal import periodogram
from scipy.stats import cauchy, entropy, norm
from sklearn.feature_extraction.text import CountVectorizer

# Manually taken from the dataset. Some words are malformed due to stemming.
CZECH_STOPWORDS = ['adsbygoogl', 'aftershar', 'api', 'appendchild', 'async', 'befor', 'btn', 'callback',
                   'clankyodjinud', 'clankyvideoportal', 'click', 'com', 'comments', 'config', 'configuration',
                   'copypast', 'count', 'createelement', 'css2', 'defaults', 'disqus', 'document', 'dsq', 'echo24cz',
                   'edita', 'elm', 'enabl', 'enabled', 'escap', 'exampl', 'fals', 'fbs', 'fjs', 'formhtml', 'forum',
                   'function', 'gatrackevents', 'gatracksocialinteractions', 'gcfg', 'getelementbyid',
                   'getelementsbytagnam', 'getjson', 'getpocket', 'getstats', 'head', 'height', 'href', 'https', 'i18n',
                   'ida', 'ihnedcz', 'insertbefor', 'into', 'javascript', 'json', 'lang', 'left', 'link', 'links',
                   'local', 'location', 'method', 'mobileoverla', 'null', 'parentnod', 'pasting', 'php', 'platform',
                   'pleas', 'pocket', 'pos', 'position', 'powered', 'ppc', 'publisherke', 'push', 'pwidget', 'queu',
                   'readmor', 'replac', 'required', 'restserver', 'return', 'rhhar0uejt6tohi9', 'sashec', 'scriptum',
                   'search', 'sharepopups', 'sharequot', 'sharer', 'shortnam', 'showerror', 'size',
                   'sklikarticleinicialized', 'sklikd', 'sklikreklam', 'src', 'success', 'the', 'titl', 'toolbar',
                   'true', 'u00edc', 'urls', 'var', 'variables', 'view', 'webpag', 'widget', 'widgets', 'width',
                   'window', 'with', 'wjs', 'writ', 'wsj', 'www', 'your', 'zoneid', 'transitioncarousel',
                   'itemloadcallback', 'initcallback', 'formatb', 'dynamiccallback', 'arrayd', 'galurl',
                   'gallerycarousel', 'galid', 'onafteranimation', 'onbeforeanimation', 'description', 'plugins',
                   'advide', 'secondtracker', 'trackingobject', 'xmlad', 'stretching', 'mous', 'navigation',
                   'translation', 'sablon', 'donation', 'mainl', 'functions', 'although', 'formatovat', 'translated',
                   'tagu', 'preddefin', 'iconk', 'formatovan', 'bbcod', 'dropdown', 'choosen']


def construct_doc_to_day_matrix(num_docs, days):
    """
    Construct a sparse binary matrix of shape (docs, days), where dtd[i, j] = 1 iff document i was published on day j.
    :param num_docs: number of documents being processed
    :param days: list of days of publication of documents, with positions corresponding to the docs list
    :return: a sparse matrix mapping documents to their publication days
    """
    doc_indices = np.arange(num_docs)
    day_indices = np.array(days)
    incidence = np.full(num_docs, 1, dtype=int)

    return sp.csc_matrix((incidence, (doc_indices, day_indices)))


def construct_feature_trajectories(bow, doc_days):
    """
    Construct a matrix of feature trajectories. Every row of the matrix represents a trajectory of a single feature.

    Trajectory of a feature f: y_f = [DFIDF(1), DFIDF(2), ..., DFIDF(T)] for T ... total number of days,
        DFIDF(t) = DF(t)/N(t) * log2(N/DF) where
            DF(t) is the number of documents containing f in day t, N(t) is the number of documents in day t,
            N is the total number of documents and DF is the total number of documents containing f.

    First, we obtain document frequency DF by counting non-zero values in BOW columns. Then, a DTD matrix mapping
    documents to their days is constructed. Counting DTD non-zero column values, we obtain N(t). By multiplying
    BOW^T @ DTD (^T indicates transpose, @ matrix multiplication), we obtain DF(t). The trajectory matrix is then
    computed as diag(log2(N/DF)) @ DF(t) @ diag(1/N(t)). This is equivalent to multiplying every column of DF(t) by
    log2(N/DF) and every row of DF(t) by 1/N(t) element-wise.

    Note: DF(t) = BOW^T @ DTD works as follows: BOW^T maps features to documents, DTD maps documents to days.
    By composing the 2 mappings, we obtain a mapping of features to days. DF(t)_ij = sum(BOW^T_ik * DTD_kj) over k
    where BOW^T_ik is the occurrence of feature i in document k, and DTD_kj is the occurrence of document k in day j.
    The result is the number of occurrences of feature i in day j.
    :param bow: bag-of-words matrix obtained from the documents
    :param doc_days: list of publication days of the document in the same order as the documents appear in bow,
        relative to the first date of the document set
    :return: matrix (features, days) of feature trajectories as a numpy array
    """
    n_samples, n_features = bow.shape
    df = bow.getnnz(axis=0).astype(float)
    idf = np.log2(float(n_samples) / df)

    t = time()
    dtd = construct_doc_to_day_matrix(n_samples, doc_days)
    n_days = dtd.shape[1]

    logging.info('Constructed DTD matrix in %fs.', time() - t)
    logging.info('DTD: %s', str(dtd.shape))

    n_t = dtd.getnnz(axis=0).astype(float)  # Number of documents in individual days.
    np.reciprocal(n_t, out=n_t)
    n_t_diag = sp.spdiags(n_t, diags=0, m=n_days, n=n_days)

    t = time()
    df_t = bow.T @ dtd  # Number of documents containing each feature in individual days.
    df_t = sp.csc_matrix(df_t, dtype=float, copy=True)

    logging.info('Multiplied BOW and DTD matrices in %fs.', time() - t)
    logging.info('DF(t): %s, %s', str(df_t.shape), str(df_t.dtype))
    logging.info('N(t): %s, %s', str(n_t.shape), str(n_t.dtype))
    logging.info('N: %d', n_samples)
    logging.info('DF: %s, %s', str(df.shape), str(df.dtype))

    t = time()

    df_t @= n_t_diag
    idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
    dfidf = idf_diag @ df_t

    logging.info('Constructed word trajectories in %fs.', time() - t)
    logging.info('DF(t)/N(t): %s, %s', str(df_t.shape), str(df_t.dtype))
    logging.info('log2(N/DF): %s, %s', str(idf.shape), str(idf.dtype))
    logging.info('DFIDF: %s, %s', str(dfidf.shape), str(dfidf.dtype))

    return dfidf.toarray()


def spectral_analysis(feature_trajectories):
    """
    Compute the periodogram, dominant power spectra (DPS) and dominant periods (DP) of the given feature trajectories.
    :param feature_trajectories: matrix of feature trajectories
    :return: DPS, DP
    """
    t = time()
    n_features, n_days = feature_trajectories.shape
    freqs, pgram = periodogram(feature_trajectories)

    with np.errstate(divide='ignore'):
        periods = np.tile(1 / freqs, (n_features, 1))

    dps_indices = np.argmax(pgram, axis=1)
    feature_indices = np.arange(n_features)

    dps = pgram[feature_indices, dps_indices]
    dp = periods[feature_indices, dps_indices]

    logging.info('Performed spectral analysis of the trajectories in %fs.', time() - t)
    logging.info('Frequencies: %s, %s', str(freqs.shape), str(freqs.dtype))
    logging.info('Periodogram: %s, %s', str(pgram.shape), str(pgram.dtype))
    logging.info('DPS: %s, %s', str(dps.shape), str(dps.dtype))
    logging.info('DP: %s, %s', str(dp.shape), str(dp.dtype))

    return dps, dp


def heuristic_stopwords_detection(feature_trajectories, dps, seed_sw_indices):
    """
    Detect stopwords in the feature set based on their DPS and average DFIDF.
    :param feature_trajectories: matrix of feature trajectories
    :param dps: dominant power spectra of the trajectories
    :param seed_sw_indices: indices of the initial stopword set within the feature trajectories
    :return: indices of all stopwords found and the DPS bound between high and low power features
    """
    t = time()
    n_features = feature_trajectories.shape[0]
    mean_dfidf = np.mean(feature_trajectories, axis=1)

    udps = np.max(dps[seed_sw_indices])
    ldfidf = np.min(mean_dfidf[seed_sw_indices])
    udfidf = np.max(mean_dfidf[seed_sw_indices])

    logging.info('UDPS: %f, LDFIDF: %f, UDFIDF: %f', udps, ldfidf, udfidf)

    stopwords_indices = []

    for i in range(n_features):
        s_fi = dps[i]
        m_dfidf = mean_dfidf[i]

        if s_fi <= udps and ldfidf <= m_dfidf <= udfidf:
            stopwords_indices.append(i)

    logging.info('Found %d stopwords in %fs.', len(stopwords_indices), time() - t)
    logging.info('Remaining features: %d', n_features - len(stopwords_indices))
    logging.info('Boundary between high and low DPS: %f', udps)

    return stopwords_indices, udps


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
        g = gmm.GaussianMixture(n_components=int(n_components), covariance_type='diag', init_params='kmeans')
        g.fit(observations)

        components = np.squeeze(np.array(
            [cauchy.pdf(days, mean[0], np.sqrt(2 * np.log(2)) * np.sqrt(cov[0])) for mean, cov in
             zip(g.means_, g.covariances_)]))

        return g.weights_ @ components

    n_features, n_days = feature_trajectories.shape

    if dominant_periods[0] > math.floor(n_days / 2):
        distributions = np.apply_along_axis(estimate_distribution_aperiodic, axis=1, arr=feature_trajectories)
    else:
        indices = np.arange(len(feature_trajectories)).reshape(-1, 1)
        distributions = np.apply_along_axis(estimate_distribution_periodic, axis=1, arr=indices)

    return {(i, j): jensen_shannon_divergence(f1, f2) for i, f1 in enumerate(distributions) for j, f2 in
            enumerate(distributions) if i < j}


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
    return max(divergences[i, j] for i in feature_indices for j in feature_indices if i < j)


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

    bow_slice = bow_matrix[:, feature_indices]
    features = feature_trajectories[feature_indices, :]
    feature_dps = dps[feature_indices]
    feature_dp = dp[feature_indices]

    return unsupervised_greedy_event_detection(feature_indices, bow_slice, features, feature_dps, feature_dp)


def create_event_trajectory(event, feature_trajectories, dps, dp):
    """
    Create a trajectory of the given event as the sum of trajectories of its features weighted by their DPS
    and normalized to probability. Also return the dominant period of the event, which is the most common
    dominant period of its features, since not all of them have necessarily the same periodicity.
    :param event: detected event (array of its feature indices)
    :param feature_trajectories: matrix of feature trajectories as row vectors
    :param dps: dominant power spectra of the processed features
    :param dp: dominant periods of the processed features
    :return: trajectory of the given event and its dominant period
    """
    e_feature_trajectories = feature_trajectories[event]
    e_power_spectra = dps[event]
    e_dominant_period = np.bincount(dp[event].astype(int)).argmax()
    e_trajectory = np.sum(e_power_spectra) * e_feature_trajectories.T @ e_power_spectra

    return e_trajectory, e_dominant_period


DPS_BOUNDARY = 0.03  # Dominant power spectrum boundary between high and low power features.
WINDOW = 7  # Length of the window to use in computing the moving average.


# TODO: Try at least a subset of the full documents, possibly using n-gram models.
# TODO: Output explained variance from LSI (see the document clustering example on sklearn webpage).
def main():
    from event_detection import data_fetchers, plotting
    total_time = time()
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    t = time()
    # documents, relative_days, weekdays = data_fetchers.fetch_signal_news(num_docs=100000)
    # documents, relative_days, weekdays = data_fetchers.fetch_de_news()
    # documents, relative_days, weekdays = data_fetchers.fetch_reuters()
    # documents, relative_days, weekdays = data_fetchers.fetch_crawl_data(num_docs=100000)
    documents, relative_days = data_fetchers.fetch_czech_corpus_dec_jan()
    # documents, relative_days = data_fetchers.fetch_czech_corpus(num_docs=10000000)

    stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.
    logging.info('Read input in %fs.', time() - t)
    logging.info('Stream length: %d', stream_length)

    t = time()
    vectorizer = CountVectorizer(min_df=30, max_df=0.9, binary=True, stop_words=CZECH_STOPWORDS)
    bow_matrix = vectorizer.fit_transform(documents)
    id2word = {v: k for k, v in vectorizer.vocabulary_.items()}
    logging.info('Created bag of words in %fs.', time() - t)
    logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype), bow_matrix.getnnz())

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
