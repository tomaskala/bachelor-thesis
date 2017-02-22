import logging
import math
from time import time

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from event_detection.k_medoids import KMedoids


class LemmatizedDocument:
    __slots__ = ['doc_id', 'document', 'similarity']

    def __init__(self, doc_id, original_document, similarity=None):
        self.doc_id = doc_id
        self.document = original_document
        self.similarity = similarity

    def __str__(self):
        return '{:d}: {:s}'.format(self.doc_id, ' '.join(self.document.name_forms))


def docids2documents(events, fetcher):
    """
    Retrieve the actual text documents for a collection events from the document IDs produced by the `postprocessing`
    module. Do this by grouping all event documents together, retrieving them in a single pass over the corpus and then
    redistributing them back to their respective events. This is to maintain some level of efficiency, since the
    lemmatized collection takes a while to iterate over.
    :param events: list of events in the format outputted by the `postprocessing.keywords2docids` functions, either
        the ones containing only document IDs or those containing both document IDs and their similarities
    :param fetcher: document fetcher to use for document streaming, should be for the same dataset as the one used for
        event detection, but can have different settings (forms instead of lemmas, different POS tags)
    :return: list of lists with each inner list representing an event and consisting of `LemmatizedDocument` objects,
        the order of the events is preserved
    """
    t = time()
    logging.info('Retrieving documents for %d events.', len(events))
    docids = []

    # Collect document IDs for all events altogether and retrieve them at once, so the collection is iterated only once.
    for event in events:
        for _, _, burst_docs in event:
            if len(burst_docs) > 0 and type(burst_docs[0]) is tuple:
                # If K-NN was used to retrieve the documents, each document is a tuple (doc_id, doc_similarity).
                docs = list(map(lambda item: item[0], burst_docs))
                docids.extend(docs)
            else:
                docids.extend(burst_docs)

    docids2docs = load_documents(docids, fetcher)
    events_out = []

    # Redistribute the documents back to the individual events, keeping similarities if they were retrieved previously.
    for event in events:
        event_out = []

        for burst_start, burst_end, burst_docs in event:
            if len(burst_docs) > 0 and type(burst_docs[0]) is tuple:
                docs_out = [LemmatizedDocument(doc_id, docids2docs[doc_id], similarity) for doc_id, similarity in
                            burst_docs]
            else:
                docs_out = [LemmatizedDocument(doc_id, docids2docs[doc_id]) for doc_id in burst_docs]

            event_out.append((burst_start, burst_end, docs_out))

        events_out.append(event_out)

    logging.info('Retrieved event documents in %fs.', time() - t)
    return events_out


def load_documents(docids, fetcher):
    """
    Load the documents with the given indices from disk.
    :param docids: IDs of documents to be loaded, will be unique-d and sorted
    :param fetcher: document fetcher to use for document streaming
    :return: dictionary mapping document IDs to the retrieved documents
    """
    if len(docids) == 0:
        raise ValueError('No document IDs given.')

    docids = list(sorted(set(docids)))
    documents = []
    doc_pos = 0

    for doc_id, document in enumerate(fetcher):
        if doc_id == docids[doc_pos]:
            documents.append(document)
            doc_pos += 1

        if doc_pos == len(docids):
            break

    return dict(zip(docids, documents))


# TODO: Read something about the parameters `a` (alpha), `l` (lambda) and `r` -- changing the value from l=6.0, r=1.0
# TODO: to l=4.0, r=0.3 (as used in the original paper) improved the summaries quite a bit.

class Summarizer:
    def __init__(self, w2v_model, min_sent_len=5, a=5.0, beta_=0.5, lambda_=4.0, r_=0.3):
        self.w2v_model = w2v_model
        self.min_sent_len = min_sent_len

        assert a >= 0, 'The parameter `a` must be >= 0'
        assert 0 <= beta_ <= 1, 'The parameter `beta` must be in [0,1]'
        assert lambda_ >= 0, 'The parameter `lambda` must be >= 0'
        assert r_ > 0, 'The parameter `r` must be > 0'

        self.a = a
        self.alpha_ = None  # `alpha` = `a` / `n_sentences`
        self.beta_ = beta_
        self.lambda_ = lambda_
        self.r_ = r_

        self.n_sentences = None
        self.n_clusters = None  # `n_clusters` = `n_sentences` // `avg_cluster_size`
        self.avg_cluster_size = 10

        self.similarity_matrix = None  # Aggregates pairwise similarities between sentences.
        self.kw_similarities = None  # Vector sentence-event_keywords similarities.
        self.cluster_similarity = None  # Pairwise similarities between sentences and clusters.

    def summarize(self, event_keywords, documents, budget, constraint_type):
        t = time()

        sentences_forms, sentences_lemma, sentences_pos = self._docs2sents(documents)
        logging.info('Created sentences in %fs.', time() - t)

        self.n_sentences = len(sentences_forms)
        self.n_clusters = self.n_sentences // self.avg_cluster_size
        self.alpha_ = self.a / self.n_sentences
        assert 0 <= self.alpha_ <= 1, 'The parameter `alpha` = `a/N` must be in [0,1]'

        t = time()
        self._precompute_similarities(event_keywords, sentences_lemma, sentences_pos)
        logging.info('Precomputed similarities in %fs.', time() - t)

        t = time()
        cluster_adjacency = self._cluster_similarities()
        logging.info('Clustered %d sentences into %d clusters in %fs.', self.n_sentences, self.n_clusters, time() - t)

        self.cluster_similarity = self.similarity_matrix @ cluster_adjacency
        del cluster_adjacency

        if constraint_type == 'words':
            constraints = np.fromiter(map(len, sentences_forms), dtype=int, count=self.n_sentences)
        elif constraint_type == 'sentences':
            constraints = np.ones(shape=self.n_sentences, dtype=int)
        else:
            raise ValueError('Invalid value for `mode`. Use either "words" or "sentences".')

        selected_indices = self._greedy_summarization(constraints, budget)
        return [sentences_forms[index] for index in selected_indices]

    def _docs2sents(self, documents):
        def is_valid_sentence(sentence_form, sentence_pos):
            has_noun = any(sent_pos[0] == 'N' for sent_pos in sentence_pos)
            has_verb = any(sent_pos[0] == 'V' for sent_pos in sentence_pos)
            long_enough = len(sentence_form) > min_sent_len
            return has_noun and has_verb and long_enough

        min_sent_len = self.min_sent_len
        used_lemmas = set()

        sentences_forms = []
        sentences_lemma = []
        sentences_pos = []

        for doc in documents:
            for form, lemma, pos in zip(doc.document.sentences_forms, doc.document.sentences_lemma,
                                        doc.document.sentences_pos):
                hashable_lemma = tuple(lemma)

                if hashable_lemma not in used_lemmas and is_valid_sentence(form, pos):
                    used_lemmas.add(hashable_lemma)
                    sentences_forms.append(form)
                    sentences_lemma.append(lemma)
                    sentences_pos.append(pos)

        return sentences_forms, sentences_lemma, sentences_pos

    def _cluster_similarities(self):
        distance_matrix = 1.0 - self.similarity_matrix

        # To avoid floating point precision errors.
        np.clip(distance_matrix, 0, 1, out=distance_matrix)
        distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0

        clusterer = KMedoids(n_clusters=self.n_clusters, distance_metric='precomputed', random_state=1)
        labels = clusterer.fit_predict(distance_matrix)

        sentence_ids = np.arange(self.n_sentences)
        cluster_adjacency = np.zeros(shape=(self.n_sentences, self.n_clusters), dtype=int)
        cluster_adjacency[sentence_ids, labels] = 1

        return cluster_adjacency

    def _precompute_similarities(self, event_keywords, sentences_lemma, sentences_pos):
        sentence_vectors = self._sents2vecs(sentences_lemma)
        self.similarity_matrix = cosine_similarity(sentence_vectors, dense_output=True)

        # Word2Vec similarity
        w2v_similarities = self._w2v_similarity(sentences_lemma)
        np.multiply(self.similarity_matrix, w2v_similarities, out=self.similarity_matrix)
        del w2v_similarities

        # LSI similarity
        lsi_similarities = self._lsi_similarity(sentence_vectors, k=50)
        np.multiply(self.similarity_matrix, lsi_similarities, out=self.similarity_matrix)
        del lsi_similarities

        # TextRank similarity
        tr_similarities = self._tr_similarity(sentences_lemma, sentences_pos)
        np.multiply(self.similarity_matrix, tr_similarities, out=self.similarity_matrix)
        del tr_similarities

        # KeyWord similarity
        kw_similarities = self._kw_similarity(sentences_lemma, event_keywords)
        np.multiply(self.similarity_matrix, kw_similarities, out=self.similarity_matrix)
        del kw_similarities

        # Transform similarities to [0,1]
        min_val = np.min(self.similarity_matrix)
        max_val = np.max(self.similarity_matrix)

        self.similarity_matrix -= min_val
        self.similarity_matrix /= (max_val - min_val)

    @staticmethod
    def _sents2vecs(sentences):
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        return tfidf.fit_transform(sentences)

    def _w2v_similarity(self, sentences):
        w2v_model = self.w2v_model
        sentence_vectors = np.empty(shape=(self.n_sentences, w2v_model.vector_size), dtype=float)

        for i, sentence in enumerate(sentences):
            sentence_vector = np.sum(
                [w2v_model[word] if word in w2v_model else np.zeros(w2v_model.vector_size, dtype=float) for word in
                 sentence], axis=0)
            sentence_vectors[i] = sentence_vector

        return (cosine_similarity(sentence_vectors, dense_output=True) + 1.0) / 2

    @staticmethod
    def _lsi_similarity(sentence_vectors, k):
        lsi = TruncatedSVD(n_components=k, random_state=1)
        lsi_matrix = lsi.fit_transform(sentence_vectors)
        return cosine_similarity(lsi_matrix, dense_output=True)

    def _tr_similarity(self, sentences, sentences_pos):
        vectorizer = CountVectorizer(binary=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc, dtype=float)

        pos_sentences = [[word for word, pos in zip(sentence, sentence_pos) if pos[0] in 'NV'] for
                         sentence, sentence_pos in zip(sentences, sentences_pos)]

        bow_matrix = vectorizer.fit_transform(pos_sentences)
        log_lengths = np.log(np.fromiter(map(len, sentences), dtype=float, count=self.n_sentences))

        sim = (bow_matrix @ bow_matrix.T) / np.add.outer(log_lengths, log_lengths)
        return sim

    def _kw_similarity(self, sentences, keywords):
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        tfidf_matrix = tfidf.fit_transform(sentences)
        kw_indices = [tfidf.vocabulary_[kw] for kw in keywords if kw in tfidf.vocabulary_]
        kw_slice = tfidf_matrix[:, kw_indices]

        self.kw_similarities = np.sum(kw_slice, axis=1)

        kw_slice_copy = kw_slice.copy()
        kw_slice_copy.data.fill(1)

        lengths = np.fromiter(map(len, sentences), dtype=float, count=self.n_sentences)

        return (kw_slice @ kw_slice_copy.T) / np.add.outer(lengths, lengths)

    def _greedy_summarization(self, constraints, budget):
        summary = []
        remainder = set(range(self.n_sentences))

        cost_so_far = 0
        objective_function = 0
        t = time()

        while len(remainder) > 0:
            k, val = self._argmax(summary, remainder, objective_function, constraints)

            if cost_so_far + constraints[k] <= budget and val - objective_function >= 0:
                cost_so_far += constraints[k]
                objective_function = val
                summary.append(k)
            # else:
            #     break  # TODO: Break or not?

            remainder.remove(k)

        singleton_candidates = set(filter(lambda singleton: constraints[singleton] <= budget, range(self.n_sentences)))
        fake_constraints = np.ones(self.n_sentences, dtype=int)
        singleton_id, singleton_val = self._argmax([], singleton_candidates, 0.0, fake_constraints)

        if singleton_val > objective_function:
            logging.info('Summarized event in %fs while filling %d out of %d budget.', time() - t,
                         constraints[singleton_id], budget)
            return [singleton_id]
        else:
            logging.info('Summarized event in %fs while filling %d out of %d budget.', time() - t, cost_so_far, budget)
            return summary

    def _argmax(self, summary, remainder, objective_function_value, constraints):
        argmax_id = None
        max_val = -math.inf
        r = self.r_

        for sentence_id in remainder:
            new_objective_value = self._quality(summary + [sentence_id])
            scaled_constraint = constraints[sentence_id] ** r

            function_gain = (new_objective_value - objective_function_value) / scaled_constraint

            if function_gain > max_val:
                argmax_id = sentence_id
                max_val = function_gain

        return argmax_id, self._quality(summary + [argmax_id])

    def _quality(self, summary):
        # return self._similarity(summary) + self.l * self._diversity(summary)
        return self._similarity(summary) + self.lambda_ * self._diversity_query(summary)

    def _similarity(self, summary):
        inter_similarity = np.sum(self.similarity_matrix[:, summary], axis=1)
        outer_similarity = np.sum(self.similarity_matrix, axis=1)

        return np.sum(np.minimum(inter_similarity, self.alpha_ * outer_similarity))

    def _diversity(self, summary):  # TODO: Which diversity to use?
        cluster_rewards = np.mean(self.cluster_similarity[summary], axis=0)
        return np.sum(np.sqrt(cluster_rewards))

    def _diversity_query(self, summary):
        summary_reward = np.mean(self.cluster_similarity[summary], axis=0)
        keyword_reward = self.kw_similarities[summary]

        return np.sum(np.sqrt(self.beta_ * summary_reward + (1 - self.beta_) * keyword_reward))
