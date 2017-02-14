import logging
import math
import os
import pickle
from time import time

import numpy as np
import scipy.sparse as sp
from hdbscan import HDBSCAN
from scipy.signal import periodogram
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer

from event_detection import annotations, data_fetchers, plotting, postprocessing, preprocessing


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


def event_detection_greedy(global_indices, doc2vec_model, feature_trajectories, dps, id2word):
    """
    The main event detection method with explicit cost function minimization.
    :param global_indices: mapping of word indices from local with respect to the examined set to global
    :param doc2vec_model: precomputed Doc2Vec model which must contain word embeddings
    :param feature_trajectories: matrix of feature trajectories
    :param dps: vector of dominant power spectra
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :return: yield the detected events as lists of keyword indices
    """

    def cost_function(old_indices, new_index):
        old_traj = np.mean(feature_trajectories[old_indices], axis=0)
        new_traj = feature_trajectories[new_index]
        trajectory_divergence = entropy(old_traj, new_traj, base=2)

        old_words = [id2word[global_indices[word_ix]] for word_ix in old_indices]
        new_word = id2word[global_indices[new_index]]
        semantic_similarity = math.exp(doc2vec_model.n_similarity(old_words, [new_word]))

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

    logging.info('Detecting events using the classical approach.')
    logging.info('Examining %d features.', len(feature_trajectories))

    # Sort feature indices by DPS in ascending order.
    indices = list(sorted(range(len(feature_trajectories)), key=lambda i: dps[i]))

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


def event_detection_cluster_based(global_indices, doc2vec_model, feature_trajectories, id2word):
    def jsd(p, q):
        """
        Compute the Jensen-Shannon divergence between the two probability distributions. Jensen-Shannon divergence is
        symmetric (in contrast to Kullback-Leibler divergence) and its square root is a proper metric.
        :param p: the true probability distribution
        :param q: the theoretical probability distribution
        :return: Jensen-Shannon divergence between the two probability distributions
        """
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
                similarity = doc2vec_model.similarity(word1, word2)
                trajectory_divergence = jsd(feature_trajectories[i], feature_trajectories[j])
                distance_matrix[i, j] = trajectory_divergence / math.exp(similarity)
                distance_matrix[j, i] = distance_matrix[i, j]

    logging.info('Precomputed word similarities in %fs.', time() - t)

    clusterer = HDBSCAN(metric='precomputed', min_samples=2, min_cluster_size=3)
    labels = clusterer.fit_predict(distance_matrix)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters == 0:
        logging.warning('No events detected.')
        return []

    events = [[] for _ in range(n_clusters)]

    for feature_ix, label in np.ndenumerate(labels):
        if label >= 0:  # Filter out the noisy samples.
            events[label].append(global_indices[feature_ix[0]])

    logging.info('Detected %d events in %fs.', len(events), time() - t)
    logging.info('Covered %d word features out of %d.', sum(len(event) for event in events), n_features)
    return events


def detect_events(doc2vec_model, feature_trajectories, dps, dp, id2word, aperiodic, cluster_based):
    """
    Perform event detection.
    :param doc2vec_model: precomputed Doc2Vec model which must contain word embeddings
    :param feature_trajectories: matrix of feature trajectories
    :param dps: vector of dominant power spectra
    :param dp: vector of dominant periods
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :param aperiodic: whether to detect aperiodic or periodic events
    :param cluster_based: whether to use the cluster-based algorithm or the one with explicit minimization
    :return: iterable of events as lists of keyword indices
    """
    n_days = feature_trajectories.shape[1]

    if aperiodic:
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp > math.floor(n_days / 2)))[0]
    else:
        feature_indices = np.where((dps > DPS_BOUNDARY) & (dp <= math.floor(n_days / 2)))[0]

    if len(feature_indices) == 0:
        logging.warning('No features to detect events from.')
        return []

    logging.info('Detecting %s events from %d features.', 'aperiodic' if aperiodic else 'periodic',
                 len(feature_indices))

    trajectories_slice = feature_trajectories[feature_indices, :]
    dps_slice = dps[feature_indices]

    if cluster_based:
        return event_detection_cluster_based(feature_indices, doc2vec_model, trajectories_slice, id2word)
    else:
        return event_detection_greedy(feature_indices, doc2vec_model, trajectories_slice, dps_slice, id2word)


def process_cluster(cluster, doc2vec_model, bow_matrix, relative_days, id2word, cluster_based, aperiodic_path,
                    periodic_path):
    """
    Detect events from a cluster of documents.
    :param cluster: indices of documents in the cluster
    :param doc2vec_model: precomputed Doc2Vec model which must contain word embeddings
    :param bow_matrix: bag-of-words matrix obtained from the documents
    :param relative_days: list of publication days of the document in the same order as the documents appear in bow,
        relative to the first date of the document set
    :param id2word: mapping from IDs to words to be used in Word2Vec
    :param cluster_based: whether to use the cluster-based algorithm or the one with explicit minimization
    :param aperiodic_path: path to store the aperiodic events to
    :param periodic_path: path to store the periodic events to
    :return: iterable of events as lists of keyword indices
    """
    trajectories = construct_feature_trajectories(bow_matrix[cluster, :], relative_days[cluster])
    dps, dp = spectral_analysis(trajectories)

    # Aperiodic events
    aperiodic_events = detect_events(doc2vec_model, trajectories, dps, dp, id2word, aperiodic=True,
                                     cluster_based=cluster_based)
    plotting.plot_events(trajectories, aperiodic_events, id2word, dps, dp, dirname=aperiodic_path)
    logging.info('Aperiodic done')

    # Periodic events
    periodic_events = detect_events(doc2vec_model, trajectories, dps, dp, id2word, aperiodic=False,
                                    cluster_based=cluster_based)
    plotting.plot_events(trajectories, periodic_events, id2word, dps, dp, dirname=periodic_path)
    logging.info('Periodic done')


# TODO: Once computed periodic events, split each event into several events whose keywords share the same periodicity.
# TODO: Or, penalize different periodicity heavily in the cost function (multiply by exp(abs(dp_difference)))?

# TODO: Putting a threshold on trajectory variation (take only std > DPS_BOUNDARY * 1.5) improves cluster based
# TODO: detection quite a bit. Do this only for the greedy approach, clusters are fine?

# TODO: Once retrieved the event documents, use Doc2Vec to put a threshold on event quality - throw away events with
# TODO: the highest similarity being < THRESHOLD (there was an event with the highest similarity being negative
# TODO: described by nonsensical words).

# Doc2Vec settings:
# 1) Concat ... Greedy OK, clusters tragic (WMD somewhat improves greedy and solves the poor cluster quality)
# 2) Mean ... Greedy OK, clusters awesome (WMD greedy are shit, but keeps the cluster quality)
# 3) Sum ... Greedy not much, clusters awesome (WMD greedy are not much, but keeps the cluster quality)
# 4) D-BOW + words ... Greedy OK, clusters tragic (WMD somewhat improves greedy and solves the poor cluster quality)

# DPS_BOUNDARY pre-clustering => 0.25, otherwise 0.05.
DPS_BOUNDARY = 0.05  # Dominant power spectrum boundary between high and low power features.
DATASET = 'dec-jan'  # Dataset is shared across all document fetchers.
# Embeddings generally need all POS tags, this removes only punctuation (Z) and unknowns (X).
POS_EMBEDDINGS = ('A', 'C', 'D', 'I', 'J', 'N', 'P', 'V', 'R', 'T')
POS_KEYWORDS = ('N', 'V', 'A', 'D')  # Allowed POS tags for keyword extraction.

PICKLE_PATH = '../event_detection/pickle'
ID2WORD_PATH = os.path.join(PICKLE_PATH, 'vectorizer_dec_jan.pickle')
BOW_MATRIX_PATH = os.path.join(PICKLE_PATH, 'term_document_dec_jan.pickle')
RELATIVE_DAYS_PATH = os.path.join(PICKLE_PATH, 'relative_days_dec_jan.pickle')

# [[(burst_start, burst_end, [(doc_id, doc_sim)] ... event_bursts] ... events] (load this to save memory)
EVENT_DOCIDS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_docids_dec_jan_greedy.pickle')
EVENT_DOCIDS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_docids_dec_jan_clusters.pickle')

# [[(burst_start, burst_end, [annotations.LemmatizedDocument] ... event_bursts] ... events]
EVENT_DOCS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_docs_dec_jan_greedy.pickle')
EVENT_DOCS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_docs_dec_jan_clusters.pickle')

# [[(burst_start, burst_end, [annotations.LemmatizedDocument] ... event_bursts] ... events]
EVENT_FULL_DOCS_GREEDY_PATH = os.path.join(PICKLE_PATH, 'event_full_docs_dec_jan_greedy.pickle')
EVENT_FULL_DOCS_CLUSTERS_PATH = os.path.join(PICKLE_PATH, 'event_full_docs_dec_jan_clusters.pickle')


def main(cluster_based, use_preclustering):
    total_time = time()
    embedding_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_EMBEDDINGS)
    keyword_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_KEYWORDS)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if os.path.exists(ID2WORD_PATH) and os.path.exists(BOW_MATRIX_PATH) and os.path.exists(RELATIVE_DAYS_PATH):
        logging.info('Deserializing id2word, bag of words matrix and relative days.')

        with open(ID2WORD_PATH, mode='rb') as f:
            id2word = pickle.load(f)

        with open(BOW_MATRIX_PATH, mode='rb') as f:
            bow_matrix = pickle.load(f)

        with open(RELATIVE_DAYS_PATH, mode='rb') as f:
            relative_days = pickle.load(f)

        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.

        logging.info('Deserialized id2word, bag of words matrix and relative days.')
        logging.info('BOW: %s, %s, storing %d elements', str(bow_matrix.shape), str(bow_matrix.dtype),
                     bow_matrix.getnnz())
        logging.info('Stream length: %d', stream_length)
    else:
        logging.info('Creating id2word, bag of words matrix and relative days.')

        if not os.path.exists(PICKLE_PATH):
            os.makedirs(PICKLE_PATH)

        t = time()
        relative_days = keyword_fetcher.fetch_relative_days()
        documents = preprocessing.LemmaPreprocessor(keyword_fetcher)

        stream_length = max(relative_days) + 1  # Zero-based, hence the + 1.
        logging.info('Read input in %fs.', time() - t)
        logging.info('Stream length: %d', stream_length)

        t = time()
        vectorizer = CountVectorizer(min_df=10, binary=True, tokenizer=lambda doc: doc, preprocessor=lambda doc: doc)
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

    num_docs = bow_matrix.shape[0]
    doc2vec_model = preprocessing.perform_doc2vec_lemma(embedding_fetcher)

    if use_preclustering:
        if cluster_based:
            aperiodic_path = './aperiodic_preclustering_clusters'
            periodic_path = './periodic_preclustering_clusters'
        else:
            aperiodic_path = './aperiodic_preclustering'
            periodic_path = './periodic_preclustering'

        document_vectors = doc2vec_model.docvecs[[i for i in range(num_docs)]]
        clusters = preprocessing.cluster_documents(document_vectors, n_clusters=6)
        del document_vectors

        for i, cluster in enumerate(clusters):
            logging.info('Processing cluster #%d.', i)
            t = time()

            process_cluster(cluster, doc2vec_model, bow_matrix, relative_days, id2word, cluster_based,
                            os.path.join(aperiodic_path, str(i)),
                            os.path.join(periodic_path, str(i)))

            logging.info('Cluster #%d processed in %fs.', i, time() - t)
    else:
        trajectories = construct_feature_trajectories(bow_matrix, relative_days)
        dps, dp = spectral_analysis(trajectories)

        if cluster_based:
            aperiodic_path = './aperiodic_clusters'
            periodic_path = './periodic_clusters'
        else:
            aperiodic_path = './aperiodic'
            periodic_path = './periodic'

        # Aperiodic events
        aperiodic_events = list(detect_events(doc2vec_model, trajectories, dps, dp, id2word, aperiodic=True,
                                              cluster_based=cluster_based))
        plotting.plot_events(trajectories, aperiodic_events, id2word, dps, dp, dirname=aperiodic_path)
        logging.info('Aperiodic done')

        # Periodic events
        periodic_events = list(detect_events(doc2vec_model, trajectories, dps, dp, id2word, aperiodic=False,
                                             cluster_based=cluster_based))
        plotting.plot_events(trajectories, periodic_events, id2word, dps, dp, dirname=periodic_path)
        logging.info('Periodic done')

        dtd = construct_doc_to_day_matrix(num_docs, relative_days)
        all_events = aperiodic_events + periodic_events

        if cluster_based:
            if os.path.exists(EVENT_DOCIDS_CLUSTERS_PATH) and os.path.exists(EVENT_DOCS_CLUSTERS_PATH):
                logging.info('Deserializing event doc IDs and documents.')

                with open(EVENT_DOCIDS_CLUSTERS_PATH, mode='rb') as f:
                    all_docids = pickle.load(f)

                with open(EVENT_DOCS_CLUSTERS_PATH, mode='rb') as f:
                    all_docs = pickle.load(f)

                logging.info('Deserialized event doc IDs and documents.')
            else:
                logging.info('Retrieving event documents.')
                t = time()

                all_docids = postprocessing.keywords2docids_wmd(keyword_fetcher, all_events, trajectories, dps, dp, dtd,
                                                                doc2vec_model, id2word)

                with open(EVENT_DOCIDS_CLUSTERS_PATH, mode='wb') as f:
                    pickle.dump(all_docids, f)

                all_docs = annotations.docids2documents(all_docids, keyword_fetcher)

                with open(EVENT_DOCS_CLUSTERS_PATH, mode='wb') as f:
                    pickle.dump(all_docs, f)

                logging.info('Retrieved and serialized event documents in %fs.', time() - t)
        else:
            if os.path.exists(EVENT_DOCIDS_GREEDY_PATH) and os.path.exists(EVENT_DOCS_GREEDY_PATH):
                logging.info('Deserializing event doc IDs and documents.')

                with open(EVENT_DOCIDS_GREEDY_PATH, mode='rb') as f:
                    all_docids = pickle.load(f)

                with open(EVENT_DOCS_GREEDY_PATH, mode='rb') as f:
                    all_docs = pickle.load(f)

                logging.info('Deserialized event doc IDs and documents.')
            else:
                logging.info('Retrieving event documents.')
                t = time()

                all_docids = postprocessing.keywords2docids_wmd(keyword_fetcher, all_events, trajectories, dps, dp, dtd,
                                                                doc2vec_model, id2word)

                with open(EVENT_DOCIDS_GREEDY_PATH, mode='wb') as f:
                    pickle.dump(all_docids, f)

                all_docs = annotations.docids2documents(all_docids, keyword_fetcher)

                with open(EVENT_DOCS_GREEDY_PATH, mode='wb') as f:
                    pickle.dump(all_docs, f)

                logging.info('Retrieved and serialized event documents in %fs.', time() - t)

        aperiodic_docs = all_docs[:len(aperiodic_events)]
        periodic_docs = all_docs[len(aperiodic_events):]

        plotting.output_events(all_events, all_docids, id2word, doc2vec_model, len(aperiodic_events), aperiodic_path,
                               periodic_path, cluster_based)
        exit()  # TODO: Don't exit

        print('Aperiodic events:', len(aperiodic_docs))

        for i, aperiodic_event in enumerate(aperiodic_docs):
            event_keywords = [id2word[keyword_ix] for keyword_ix in aperiodic_events[i]]
            print('Aperiodic event {}: [{}]'.format(i, ', '.join(event_keywords)))

            for burst in aperiodic_event:
                burst_start, burst_end, burst_docs = burst
                print('Burst ({} - {}): {} docs'.format(burst_start, burst_end, len(burst_docs)))
                print('Most similar:', burst_docs[0], burst_docs[0].similarity)
                print('Least similar', burst_docs[-1], burst_docs[-1].similarity)

            print()

        print('Periodic events:', len(periodic_docs))

        for i, periodic_event in enumerate(periodic_docs):
            event_keywords = [id2word[keyword_ix] for keyword_ix in periodic_events[i]]
            print('Periodic event {}: [{}]'.format(i, ', '.join(event_keywords)))

            for burst in periodic_event:
                burst_start, burst_end, burst_docs = burst
                print('Burst ({} - {}): {} docs'.format(burst_start, burst_end, len(burst_docs)))
                print('Most similar:', burst_docs[0], burst_docs[0].similarity)
                print('Least similar', burst_docs[-1], burst_docs[-1].similarity)

            print()

    logging.info('All done in %fs.', time() - total_time)


if __name__ == '__main__':
    main(cluster_based=True, use_preclustering=False)
