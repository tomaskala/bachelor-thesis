import logging
import math
import re
from time import time

import numpy as np
from sklearn.preprocessing import normalize

from event_detection import sphere


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


class Summarizer:
    def __init__(self, w2v_model, a=0.75, l=6.0, r=1.0):
        self.w2v_model = w2v_model

        assert 0 <= a <= 1, 'The parameter `a` must be in [0,1]'
        assert l >= 0, 'The parameter `l` must be >= 0'
        assert r > 0, 'The parameter `r` must be > 0'

        self.a = a
        self.l = l
        self.r = r
        self.similarity_matrix = None
        self.cluster_similarity = None

    def summarize(self, documents, budget, constraint_type):
        t = time()
        sentences_forms, sentences_lemma = self._docs2sents(documents)
        logging.info('Created sentences in %fs.', time() - t)

        n = len(sentences_forms)
        k = n // 5

        t = time()
        sentence_vectors = self._sents2vecs(sentences_lemma)
        logging.info('Created sentence vectors in %fs.', time() - t)

        t = time()
        cluster_representation = self._cluster_sentences(sentence_vectors, k)
        logging.info('Performed sentence clustering in %fs.', time() - t)

        t = time()
        self._precompute_similarities(sentence_vectors)
        logging.info('Precomputed similarities in %fs.', time() - t)

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

    @staticmethod
    def _docs2sents(documents):
        used_lemmas = set()
        sentences_forms = []
        sentences_lemma = []

        for doc in documents:
            for form, lemma in zip(doc.document.sentences_forms, doc.document.sentences_lemma):
                hashable_lemma = tuple(lemma)

                if hashable_lemma not in used_lemmas:
                    used_lemmas.add(hashable_lemma)
                    sentences_forms.append(form)
                    sentences_lemma.append(lemma)

        return sentences_forms, sentences_lemma

    def _sents2vecs(self, sentences):
        w2v_model = self.w2v_model
        sentence_vectors = np.empty(shape=(len(sentences), w2v_model.vector_size), dtype=float)

        for i, sentence in enumerate(sentences):
            sentence_vector = np.mean(
                [w2v_model[word] if word in w2v_model else np.zeros(w2v_model.vector_size, dtype=float) for word in
                 sentence], axis=0)
            sentence_vectors[i] = sentence_vector

        normalize(sentence_vectors, copy=False)
        return sentence_vectors

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

    def _precompute_similarities(self, sentence_vectors):
        self.similarity_matrix = sentence_vectors @ sentence_vectors.T

        # Transform the similarities into from [-1,1] to [0,1].
        self.similarity_matrix += 1
        self.similarity_matrix /= 2

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


def main():
    from event_detection import data_fetchers
    events = [[(2, 3, [(0, 100), (2, 102), (3, 103)]), (4, 7, [(5, 205), (9, 209), (13, 213)])],
              [(1, 8, [(1, 301), (4, 304), (2, 302)]), (2, 3, [(0, 400), (2, 402), (3, 403)])],
              [(3, 6, [(8, 508), (5, 505), (2, 502)])]]
    new_events = docids2documents(events, data_fetchers.CzechLemmatizedTexts(dataset='dec-jan', fetch_forms=True))

    for i, event in enumerate(new_events):
        print('Event {:d}'.format(i))

        for burst_start, burst_end, burst_docs in event:
            print('Burst: ({:d}, {:d})'.format(burst_start, burst_end))

            for doc in burst_docs:
                print(doc, doc.similarity)

        print()


if __name__ == '__main__':
    main()
