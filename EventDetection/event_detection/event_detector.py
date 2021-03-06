import logging
import math
import os
import pickle
from datetime import datetime, timedelta
from time import time

import numpy as np
import scipy.sparse as sp
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

import gensim

from event_detection import annotations, data_fetchers, plotting, postprocessing, preprocessing


def construct_doc_to_day_matrix(n_docs, relative_days, names_separately=False):
    """
    Construct a sparse binary matrix of shape (docs, days), where dtd[i, j] = 1 iff document i was published on day j.
    :param n_docs: number of documents being processed
    :param relative_days: list of days of publication of documents, with positions corresponding to the docs list
    :param names_separately: whether the documents in BOW matrix are in pairs (document_headline, document_body) and we
        only want indices of real documents with respect to the document collection, not these 'pseudo-documents' which
        the real documents are divided into as names and bodies
    :return: a sparse matrix mapping documents to their publication days
    """
    if names_separately:
        n_docs //= 2
        relative_days = relative_days[::2]

    doc_indices = np.arange(n_docs)
    day_indices = np.array(relative_days)
    incidence = np.full(n_docs, 1, dtype=int)

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


def set_similarity(feature_indices, divergences):
    """
    Compute the similarity of a set of features using the precomputed KL-divergences. Set similarity is defined as
    the maximum of divergences between all pairs of features from the set.
    :param feature_indices: indices of the features from the set
    :param divergences: precomputed matrix of feature trajectory divergences
    :return: similarity of the set
    """
    return np.max(divergences[np.ix_(feature_indices, feature_indices)])


def event_detection_greedy(global_indices, w2v_model, feature_trajectories, dps, id2word):
    """
    The main event detection method with explicit cost function minimization.
    :param global_indices: mapping of word indices from local with respect to the examined set to global
    :param w2v_model: trained Word2Vec model
    :param feature_trajectories: matrix of feature trajectories
    :param dps: vector of dominant power spectra
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :return: the detected events as a list of arrays of keyword indices
    """

    def cost_function(old_indices, new_index):
        trajectory_divergence = set_similarity(old_indices + [new_index], similarities)

        old_words = [id2word[global_indices[word_ix]] for word_ix in old_indices]
        new_word = id2word[global_indices[new_index]]
        semantic_similarity = (w2v_model.n_similarity(old_words, [new_word]) + 1) / 2

        dps_score = np.sum(dps[old_indices + [new_index]])
        return trajectory_divergence / (semantic_similarity * dps_score)

    def minimizing_feature(event_so_far, feature_indices):
        index = feature_indices[0]
        min_cost = cost_function(event_so_far, feature_indices[0])

        for f in feature_indices[1:]:
            added_cost = cost_function(event_so_far, f)

            if added_cost < min_cost:
                index, min_cost = f, added_cost

        return index, min_cost

    logging.info('Detecting events using the greedy approach.')
    logging.info('Examining %d features.', len(feature_trajectories))

    # Sort feature indices by DPS in descending order.
    indices = list(sorted(range(len(feature_trajectories)), key=lambda k: dps[k], reverse=True))
    n_features = feature_trajectories.shape[0]

    similarities = np.zeros((n_features, n_features), dtype=float)

    for i, f1 in enumerate(feature_trajectories):
        for j, f2 in enumerate(feature_trajectories):
            if i < j:
                similarities[i, j] = max(entropy(f1, f2, base=2), entropy(f2, f1, base=2))
                similarities[j, i] = similarities[i, j]

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


def event_detection_cluster_based(global_indices, w2v_model, feature_trajectories, id2word):
    """
    The main event detection method with clustering.
    :param global_indices: mapping of word indices from local with respect to the examined set to global
    :param w2v_model: trained Word2Vec model
    :param feature_trajectories: matrix of feature trajectories
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :return: the detected events as a list of arrays of keyword indices
    """

    def jsd(p, q):
        m = 0.5 * (p + q)
        return 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    n_features = len(global_indices)
    logging.info('Detecting events using the clustering approach.')
    logging.info('Examining %d features.', n_features)
    t = time()

    distance_matrix = np.zeros((n_features, n_features), dtype=float)

    for i in range(n_features):
        for j in range(n_features):
            if i > j:
                word1 = id2word[global_indices[i]]
                word2 = id2word[global_indices[j]]

                try:
                    word1 = gensim.parsing.preprocessing.strip_punctuation(word1).split()[0]
                    word2 = gensim.parsing.preprocessing.strip_punctuation(word2).split()[0]

                    vec1 = w2v_model[word1]
                    vec2 = w2v_model[word2]
                    dist = np.linalg.norm(vec1 - vec2)

                    trajectory_divergence = jsd(feature_trajectories[i], feature_trajectories[j])
                    distance_matrix[i, j] = trajectory_divergence * dist
                    distance_matrix[j, i] = distance_matrix[i, j]
                except IndexError:
                    distance_matrix[i, j] = 1e6
                    distance_matrix[j, i] = 1e6

    logging.info('Precomputed word similarities in %fs.', time() - t)

    distance_matrix[distance_matrix > 1e6] = 1e6
    distance_matrix[np.isnan(distance_matrix)] = 1e6

    clusterer = DBSCAN(metric='precomputed', min_samples=2)
    labels = clusterer.fit_predict(distance_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 0:
        logging.warning('No events detected.')
        return []

    events = [[] for _ in range(n_clusters)]

    for feature_ix, label in enumerate(labels):
        if label >= 0:  # Filter out the noisy samples.
            events[label].append(global_indices[feature_ix])

    logging.info('Detected %d events in %fs.', len(events), time() - t)
    logging.info('Covered %d word features out of %d.', sum(len(event) for event in events), n_features)
    logging.info('Silhouette score: %f.', silhouette_score(distance_matrix, labels, metric='precomputed'))
    logging.info('Cluster sizes: %s.', str([len(event) for event in events]))
    return events


def rolling_window(array, window):
    """
    Calculate a rolling window of the given array along the -1st axis, similar to convolving each row with
    `np.ones(window) / window`, but implemented more efficiently.
    :param array: the array along whose rows to compute the rolling window
    :param window: rolling window size
    :return: array of row rolling windows of the given array
    """
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


def calculate_trajectory_cutoff(trajectories, window):
    """
    Calculate a vector of cutoff values, one for each given trajectory. Trajectory elements falling under each cutoff
    can be safely ignored as they are most likely caused by noise and do not contribute to any major event.
    :param trajectories: matrix of trajectories
    :param window: window size for moving average
    :return: vector of cutoff values whose size is the number of trajectories
    """
    ma = np.mean(rolling_window(trajectories, window), -1)
    ma_mean = np.mean(ma, axis=1)
    ma_std = np.std(ma, axis=1)
    cutoff = ma_mean + ma_std

    return cutoff.reshape(-1, 1)


def detect_events(w2v_model, feature_trajectories, dps, dp, id2word, which, cluster_based):
    """
    Perform event detection.
    :param w2v_model: trained Word2Vec model
    :param feature_trajectories: matrix of feature trajectories
    :param dps: vector of dominant power spectra
    :param dp: vector of dominant periods
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :param which: whether to detect 'aperiodic', 'periodic' or 'all' events
    :param cluster_based: whether to use the cluster-based algorithm or the one with explicit minimization
    :return: iterable of events as lists of keyword indices
    """
    n_days = feature_trajectories.shape[1]

    if which == 'aperiodic':
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp > math.floor(n_days / 2)))[0]
    elif which == 'periodic':
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp <= math.floor(n_days / 2)))[0]
    else:
        feature_indices = np.where(dps > DPS_BOUNDARY)[0]

    if len(feature_indices) == 0:
        logging.warning('No features to detect events from.')
        return []

    if cluster_based:
        # Greedy approach performs poorly when noise is trimmed, but clusters thrive.
        cutoff = calculate_trajectory_cutoff(feature_trajectories, WINDOW)
        feature_trajectories[feature_trajectories <= cutoff] = 0.0

    logging.info('Detecting %s events from %d features.', which, len(feature_indices))

    trajectories_slice = feature_trajectories[feature_indices]
    dps_slice = dps[feature_indices]

    if cluster_based:
        return event_detection_cluster_based(feature_indices, w2v_model, trajectories_slice, id2word)
    else:
        return list(filter(lambda event: len(event) > 2,
                           event_detection_greedy(feature_indices, w2v_model, trajectories_slice, dps_slice, id2word)))


def summarize_events(events, events_docids_repr, id2word, w2v_model, cluster_based):
    """
    Perform multi-document summarization on documents retrieved for the events detected. The summaries will be
    output to a file based on the `cluster_based` parameter.
    :param events: the detected events
    :param events_docids_repr: document ID representation of the events retrieved using functions in `postprocessing.py`
    :param id2word: mapping from IDs to words
    :param w2v_model: trained Word2Vec model
    :param cluster_based: whether to use the cluster-based algorithm or the one with explicit minimization
    """
    summarization_fetcher = data_fetchers.CzechSummarizationTexts(dataset=DATASET)

    if cluster_based:
        if os.path.exists(EVENT_SUMM_DOCS_CLUSTERS_PATH):
            logging.info('Deserializing full documents.')

            with open(EVENT_SUMM_DOCS_CLUSTERS_PATH, mode='rb') as f:
                events_docs_repr = pickle.load(f)

            logging.info('Deserialized full documents.')
        else:
            logging.info('Retrieving full documents.')
            t = time()

            events_docs_repr = annotations.docids2documents(events_docids_repr, summarization_fetcher)

            with open(EVENT_SUMM_DOCS_CLUSTERS_PATH, mode='wb') as f:
                pickle.dump(events_docs_repr, f)

            logging.info('Retrieved and serialized full documents in %fs.', time() - t)
    else:
        if os.path.exists(EVENT_SUMM_DOCS_GREEDY_PATH):
            logging.info('Deserializing full documents.')

            with open(EVENT_SUMM_DOCS_GREEDY_PATH, mode='rb') as f:
                events_docs_repr = pickle.load(f)

            logging.info('Deserialized full documents.')
        else:
            logging.info('Retrieving full documents.')
            t = time()

            events_docs_repr = annotations.docids2documents(events_docids_repr, summarization_fetcher)

            with open(EVENT_SUMM_DOCS_GREEDY_PATH, mode='wb') as f:
                pickle.dump(events_docs_repr, f)

            logging.info('Retrieved and serialized full documents in %fs.', time() - t)

    file_path = './cluster_summaries.txt' if cluster_based else './greedy_summaries.txt'
    summarize_inner(events_docs_repr, events, id2word, w2v_model, file_path)


def summarize_inner(events_docs_repr, events, id2word, w2v_model, file_path):
    """
    Do the actual summarization with the retrieved documents.
    :param events_docs_repr: document representation of the events retrieved using functions in `postprocessing.py`
    :param events: the detected events
    :param id2word: mapping from IDs to words
    :param w2v_model: trained Word2Vec model
    :param file_path: where to store the summaries
    """
    summarizer = annotations.Summarizer(w2v_model)
    constraint_type = 'words'
    budget = 60

    t = time()

    with open(file_path, 'w', encoding='utf8') as f:
        for i, event in enumerate(events_docs_repr):
            event_keywords = [id2word[keyword_id] for keyword_id in events[i]]

            if len(event_keywords) <= 8:
                keywords_out = ', '.join(event_keywords)
            else:
                keywords_out = ', '.join(event_keywords[:8]) + '...'

            event_out = '{:d} - {:s}\n'.format(i + 1, keywords_out)

            for burst in event:
                burst_start, burst_end, burst_docs = burst

                burst_start_date = datetime(year=2014, month=1, day=1) + timedelta(days=burst_start)
                burst_end_date = datetime(year=2014, month=1, day=1) + timedelta(days=burst_end)

                burst_start_out = f'{burst_start_date:%b} {burst_start_date.day}'
                burst_end_out = f'{burst_end_date:%b} {burst_end_date.day}, {burst_end_date:%Y}'

                if len(burst_docs) == 0:
                    event_out += 'burst {:s} - {:s} CONTAINS NO DOCUMENTS\n'.format(burst_start_out, burst_end_out)
                    continue

                sentences = summarizer.summarize(event_keywords, burst_docs[:50], budget, constraint_type)
                sentences_out = ' '.join(map(lambda sentence: ' '.join(sentence), sentences))
                headline_out = ' '.join(burst_docs[0].document.name_forms)
                event_out += 'burst {:s} - {:s}\n{:s}\n{:s}\n'.format(burst_start_out, burst_end_out, headline_out,
                                                                      sentences_out)

            print(event_out, file=f)
            print(file=f)

    print('Summarized the events in {:f}s.'.format(time() - t))


DPS_BOUNDARY = 0.75  # Boundary between eventful and non-eventful words.
DATASET = 'full'  # Dataset is shared across all document fetchers.
# Embeddings generally need all POS tags, this removes only punctuation (Z) and unknowns (X).
POS_EMBEDDINGS = ('A', 'C', 'D', 'I', 'J', 'N', 'P', 'V', 'R', 'T')
POS_KEYWORDS = ('N', 'V', 'A', 'D')  # Allowed POS tags for keyword extraction.
NAMES_SEPARATELY = True  # Whether the documents are pairs (document_headline, document_body) for each real document.
WINDOW = 7  # Window size for moving average.

PICKLE_PATH = '../event_detection/pickle'
ID2WORD_PATH = os.path.join(PICKLE_PATH, 'id2word.pickle')
BOW_MATRIX_PATH = os.path.join(PICKLE_PATH, 'term_document.npz')
RELATIVE_DAYS_PATH = os.path.join(PICKLE_PATH, 'relative_days.pickle')

# [[(burst_start, burst_end, [(doc_id, doc_sim)] ... event_bursts] ... events]
EVENT_DOCIDS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_docids_greedy.pickle')
EVENT_DOCIDS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_docids_clusters.pickle')

# [[(burst_start, burst_end, [annotations.LemmatizedDocument] ... event_bursts] ... events]
EVENT_SUMM_DOCS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_greedy.pickle')
EVENT_SUMM_DOCS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_summ_docs_clusters.pickle')


def main(cluster_based):
    global DPS_BOUNDARY

    if cluster_based:
        DPS_BOUNDARY = 0.01
    else:
        DPS_BOUNDARY = 0.04

    total_time = time()
    embedding_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_EMBEDDINGS)
    keyword_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_KEYWORDS)

    # Step 1: Construct and analyze feature trajectories.
    # ---------------------------------------------------

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

    n_docs = bow_matrix.shape[0]
    trajectories = construct_feature_trajectories(bow_matrix, relative_days)

    # Bag of Words matrix is no longer needed -> delete it to free some memory.
    del bow_matrix

    dps, dp = postprocessing.spectral_analysis(trajectories)
    w2v_model = preprocessing.perform_word2vec(embedding_fetcher, NAMES_SEPARATELY)

    # Step 2: Detect events
    # ---------------------

    # All events
    events = detect_events(w2v_model, trajectories, dps, dp, id2word, which='all', cluster_based=cluster_based)

    # Step 2.5: Discard low period events
    # -----------------------------------
    event_trajectories, event_periods = postprocessing.create_events_trajectories(events, trajectories, dps)
    kept_events = []

    for event, period in zip(events, event_periods):
        if period > 7:
            kept_events.append(event)

    events = kept_events
    del kept_events
    del event_trajectories
    del event_periods

    plotting.plot_events(trajectories, events, id2word, dps,
                         dirname='./events_clusters' if cluster_based else './events_greedy')
    logging.info('Events detected')

    # Step 3: Obtain IDs of documents related to each event.
    # ------------------------------------------------------

    dtd = construct_doc_to_day_matrix(n_docs, relative_days, names_separately=NAMES_SEPARATELY)

    # Relative days are no longer needed -> delete them to free some memory.
    del relative_days

    if cluster_based:
        with open(os.path.join(PICKLE_PATH, 'events_clusters.pickle'), mode='rb') as f:
            events = pickle.load(f)

        if os.path.exists(EVENT_DOCIDS_CLUSTERS_PATH):
            logging.info('Deserializing event doc IDs.')

            with open(EVENT_DOCIDS_CLUSTERS_PATH, mode='rb') as f:
                all_docids = pickle.load(f)

            logging.info('Deserialized event doc IDs.')
        else:
            logging.info('Retrieving event doc IDs.')
            t = time()

            all_docids = postprocessing.keywords2docids_wmd(keyword_fetcher, events, trajectories, dps, dtd,
                                                            w2v_model, id2word)

            with open(EVENT_DOCIDS_CLUSTERS_PATH, mode='wb') as f:
                pickle.dump(all_docids, f)

            logging.info('Retrieved and serialized event doc IDs in %fs.', time() - t)
    else:
        if os.path.exists(EVENT_DOCIDS_GREEDY_PATH):
            logging.info('Deserializing event doc IDs.')

            with open(EVENT_DOCIDS_GREEDY_PATH, mode='rb') as f:
                all_docids = pickle.load(f)

            logging.info('Deserialized event doc IDs.')
        else:
            logging.info('Retrieving event doc IDs.')
            t = time()

            all_docids = postprocessing.keywords2docids_wmd(keyword_fetcher, events, trajectories, dps, dtd,
                                                            w2v_model, id2word)

            with open(EVENT_DOCIDS_GREEDY_PATH, mode='wb') as f:
                pickle.dump(all_docids, f)

            logging.info('Retrieved and serialized event doc IDs in %fs.', time() - t)

    # Step 4: Summarize the events.
    # -----------------------------

    summarize_events(events, all_docids, id2word, w2v_model, cluster_based)
    logging.info('All done in %fs.', time() - total_time)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main(cluster_based=False)
