import logging
import math
from time import time

import gensim
import numpy as np
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


# TODO: (http://publications.lib.chalmers.se/records/fulltext/174136/174136.pdf)
# TODO: Title boosting - if a sentence contains a named entity from the title, boost its importance

class Summarizer:
    def __init__(self, w2v_model, min_sent_len=5, a=0.75, l=6.0, r=1.0):
        self.w2v_model = w2v_model
        self.min_sent_len = min_sent_len

        assert 0 <= a <= 1, 'The parameter `a` must be in [0,1]'
        assert l >= 0, 'The parameter `l` must be >= 0'
        assert r > 0, 'The parameter `r` must be > 0'

        self.a = a
        self.l = l
        self.r = r
        self.similarity_matrix = None
        self.cluster_similarity = None

    def summarize(self, event_keywords, documents, budget, constraint_type):
        t = time()
        sentences_forms, sentences_lemma, sentences_pos = self._docs2sents(documents)
        logging.info('Created sentences in %fs.', time() - t)

        n = len(sentences_forms)
        k = n // 5

        t = time()
        sentence_vectors = self._sents2vecs(sentences_lemma)
        logging.info('Created sentence vectors in %fs.', time() - t)

        # t = time()
        # cluster_representation = self._cluster_sentences(sentence_vectors, k)
        # logging.info('Performed sentence clustering in %fs.', time() - t)

        t = time()
        self._precompute_similarities(event_keywords, sentence_vectors, sentences_lemma, sentences_pos)
        logging.info('Precomputed similarities in %fs.', time() - t)

        t = time()
        cluster_representation = self._cluster_similarities(k)
        logging.info('Performed sentence clustering in %fs.', time() - t)

        del sentence_vectors
        self.cluster_similarity = self.similarity_matrix @ cluster_representation
        del cluster_representation

        if constraint_type == 'words':
            constraints = np.fromiter(map(len, sentences_forms), dtype=int, count=n)
        elif constraint_type == 'sentences':
            constraints = np.ones(shape=n, dtype=int)
        else:
            raise ValueError('Invalid value for `mode`. Use either "words" or "sentences".')

        selected_indices = self._greedy_summarization(constraints, budget)
        return [sentences_forms[index] for index in selected_indices]

    def _docs2sents(self, documents):
        used_lemmas = set()
        sentences_forms = []
        sentences_lemma = []
        sentences_pos = []
        min_sent_len = self.min_sent_len

        for doc in documents:
            for form, lemma, pos in zip(doc.document.sentences_forms, doc.document.sentences_lemma,
                                        doc.document.sentences_pos):
                hashable_lemma = tuple(lemma)

                if len(form) > min_sent_len and hashable_lemma not in used_lemmas:
                    used_lemmas.add(hashable_lemma)
                    sentences_forms.append(form)
                    sentences_lemma.append(lemma)
                    sentences_pos.append(pos)

        return sentences_forms, sentences_lemma, sentences_pos

    def _sents2vecs(self, sentences):
        # w2v_model = self.w2v_model
        # sentence_vectors = np.empty(shape=(len(sentences), w2v_model.vector_size), dtype=float)
        #
        # for i, sentence in enumerate(sentences):
        #     sentence_vector = np.mean(
        #         [w2v_model[word] if word in w2v_model else np.zeros(w2v_model.vector_size, dtype=float) for word in
        #          sentence], axis=0)
        #     sentence_vectors[i] = sentence_vector
        #
        # normalize(sentence_vectors, copy=False)
        # return sentence_vectors
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        return tfidf.fit_transform(sentences)

    @staticmethod
    def _cluster_sentences(sentence_vectors, k):
        from sklearn.cluster import MiniBatchKMeans
        # clusterer = sphere.SphericalKMeans(n_clusters=k, random_state=1)
        clusterer = MiniBatchKMeans(n_clusters=k, random_state=1)
        labels = clusterer.fit_predict(sentence_vectors)

        n_sentences = sentence_vectors.shape[0]
        observations = np.arange(n_sentences)
        cluster_representation = np.zeros(shape=(n_sentences, k), dtype=int)

        cluster_representation[observations, labels] = 1

        logging.info('Clustered %d sentences into %d clusters.', n_sentences, k)

        return cluster_representation

    def _cluster_similarities(self, k):
        distance_matrix = 1.0 - self.similarity_matrix
        np.clip(distance_matrix, 0, 1, out=distance_matrix)
        distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0

        clusterer = KMedoids(n_clusters=k, distance_metric='precomputed')
        labels = clusterer.fit_predict(distance_matrix)

        n_sentences = self.similarity_matrix.shape[0]
        observations = np.arange(n_sentences)
        cluster_representation = np.zeros(shape=(n_sentences, k), dtype=int)

        cluster_representation[observations, labels] = 1

        logging.info('Clustered %d sentences into %d clusters.', n_sentences, k)

        return cluster_representation

    def _precompute_similarities(self, event_keywords, sentence_vectors, sentences, sentences_pos):
        # self.similarity_matrix = sentence_vectors @ sentence_vectors.T

        # Transform the similarities into from [-1,1] to [0,1].
        # self.similarity_matrix += 1
        # self.similarity_matrix /= 2
        self.similarity_matrix = cosine_similarity(sentence_vectors, dense_output=True)

        w2v_similarities = self._w2v_similarity(sentences)
        np.multiply(self.similarity_matrix, w2v_similarities, out=self.similarity_matrix)
        del w2v_similarities

        # d2v_similarities = self._d2v_similarity(sentences)
        # np.multiply(self.similarity_matrix, d2v_similarities, out=self.similarity_matrix)
        # del d2v_similarities

        lsi_similarities = self._lsi_similarity(sentence_vectors, k=50)
        np.multiply(self.similarity_matrix, lsi_similarities, out=self.similarity_matrix)
        del lsi_similarities

        tr_similarities = self._tr_similarity(sentences, sentences_pos)
        np.multiply(self.similarity_matrix, tr_similarities, out=self.similarity_matrix)
        del tr_similarities

        kw_similarities = self._kw_similarity(sentences, event_keywords)
        np.multiply(self.similarity_matrix, kw_similarities, out=self.similarity_matrix)
        del kw_similarities

        min_val = np.min(self.similarity_matrix)
        max_val = np.max(self.similarity_matrix)

        self.similarity_matrix -= min_val
        self.similarity_matrix /= (max_val - min_val)

    def _w2v_similarity(self, sentences):
        w2v_model = self.w2v_model
        sentence_vectors = np.empty(shape=(len(sentences), w2v_model.vector_size), dtype=float)

        for i, sentence in enumerate(sentences):
            sentence_vector = np.sum(
                [w2v_model[word] if word in w2v_model else np.zeros(w2v_model.vector_size, dtype=float) for word in
                 sentence], axis=0)
            sentence_vectors[i] = sentence_vector

        return cosine_similarity(sentence_vectors, dense_output=True)

    def _d2v_similarity(self, sentences):
        class Tagger:
            def __init__(self, sentences_):
                self.sentences_ = sentences_

            def __iter__(self):
                for i, sentence_ in enumerate(self.sentences_):
                    tags = [i]
                    words = [word for word in sentence_ if len(word) > 1]
                    tagged_doc = gensim.models.doc2vec.TaggedDocument(words, tags)

                    yield tagged_doc

        tagger = Tagger(sentences)
        size = 20
        logging.disable(logging.WARNING)  # Gensim loggers are super chatty.
        d2v_model = gensim.models.Doc2Vec(tagger, dm=1, dm_concat=0, dm_mean=1, size=size, batch_words=1000)
        logging.disable(logging.NOTSET)  # Re-enable logging.

        sentence_vectors = np.empty(shape=(len(sentences), size), dtype=float)

        for j, sentence in enumerate(sentences):
            sentence_vector = d2v_model.docvecs[j]
            sentence_vectors[j] = sentence_vector

        return cosine_similarity(sentence_vectors, dense_output=True)

    def _wmd_similarity(self, sentences):
        logging.disable(logging.INFO)  # Gensim loggers are super chatty.
        n_sentences = len(sentences)
        w2v_model = self.w2v_model
        wmd_similarities = np.zeros((n_sentences, n_sentences), dtype=float)

        for i in range(n_sentences):
            for j in range(n_sentences):
                if i < j:
                    wmd_similarities[i, j] = w2v_model.wmdistance(sentences[i], sentences[j])

        wmd_similarities += wmd_similarities.T
        wmd_similarities += 1.0
        np.reciprocal(wmd_similarities, out=wmd_similarities)

        logging.disable(logging.NOTSET)  # Re-enable logging.
        return wmd_similarities

    @staticmethod
    def _lsi_similarity(sentence_vectors, k):
        from sklearn.decomposition import TruncatedSVD
        lsi = TruncatedSVD(n_components=k)
        lsi_matrix = lsi.fit_transform(sentence_vectors)
        return cosine_similarity(lsi_matrix, dense_output=True)

    @staticmethod
    def _tr_similarity(sentences, sentences_pos):
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(binary=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc, dtype=float)

        pos_sentences = [[word for word, pos in zip(sentence, sentence_pos) if pos[0] in 'NV'] for
                         sentence, sentence_pos in zip(sentences, sentences_pos)]

        bow_matrix = vectorizer.fit_transform(pos_sentences)
        log_lengths = np.log(np.fromiter(map(len, sentences), dtype=float, count=len(sentences)))

        sim = (bow_matrix @ bow_matrix.T) / np.add.outer(log_lengths, log_lengths)
        # np.clip(sim, 0, 1, out=sim)  # TODO: Put this here or not?
        return sim

    @staticmethod
    def _kw_similarity(sentences, keywords):
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        tfidf_matrix = tfidf.fit_transform(sentences)
        kw_indices = [tfidf.vocabulary_[kw] for kw in keywords if kw in tfidf.vocabulary_]
        kw_slice = tfidf_matrix[:, kw_indices]

        kw_slice_copy = kw_slice.copy()
        kw_slice_copy.data.fill(1)

        lengths = np.fromiter(map(len, sentences), dtype=float, count=len(sentences))

        return (kw_slice @ kw_slice_copy.T) / np.add.outer(lengths, lengths)

    def _greedy_summarization(self, constraints, budget):
        n_sentences = self.similarity_matrix.shape[0]
        summary = []
        remainder = set(range(n_sentences))

        cost_so_far = 0
        objective_function = 0

        while len(remainder) > 0:
            k, val = self._argmax(summary, remainder, objective_function, constraints)

            if cost_so_far + constraints[k] <= budget and val - objective_function >= 0:
                cost_so_far += constraints[k]
                objective_function = val
                summary.append(k)

            remainder.remove(k)

        singleton_candidates = set(filter(lambda singleton: constraints[singleton] <= budget, range(n_sentences)))
        fake_constraints = np.ones(n_sentences, dtype=int)
        singleton_ix, singleton_val = self._argmax([], singleton_candidates, 0.0, fake_constraints)

        if singleton_val > objective_function:
            return [singleton_ix]
        else:
            return summary

    def _argmax(self, summary, remainder, objective_function_value, constraints):
        argmax_ix = None
        max_val = -math.inf
        r = self.r

        # t = time()

        for i, sentence_ix in enumerate(remainder):
            new_objective_value = self._quality(summary + [sentence_ix])
            scaled_constraint = constraints[sentence_ix] ** r

            function_gain = (new_objective_value - objective_function_value) / scaled_constraint

            if function_gain > max_val:
                argmax_ix = sentence_ix
                max_val = function_gain

                # if i > 0 and i % 100 == 0:
                #     logging.info('Performed 100 argmax iterations in %fs.', time() - t)
                #     t = time()

        return argmax_ix, self._quality(summary + [argmax_ix])

    def _quality(self, summary):
        return self._similarity(summary) + self.l * self._diversity(summary)

    def _similarity(self, summary):
        similarity_matrix = self.similarity_matrix

        inter_similarity = np.sum(similarity_matrix[:, summary], axis=1)
        outer_similarity = np.sum(similarity_matrix, axis=1)

        return np.sum(np.minimum(inter_similarity, self.a * outer_similarity))

    def _diversity(self, summary):
        cluster_rewards = np.mean(self.cluster_similarity[summary], axis=0)
        return np.sum(np.sqrt(cluster_rewards))

    def __str__(self):
        return 'Summarizer(W2W: {:s}, alpha: {:f}, lambda: {:f}, r: {:f})'.format(str(self.w2v_model), self.a, self.l,
                                                                                  self.r)
