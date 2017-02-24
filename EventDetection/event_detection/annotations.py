import csv
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


class Summarizer:
    """
    Based on 'Lin, Bilmes, 2010, Multi-document Summarization via Budgeted Maximization of Submodular Functions'
    and 'Mogren, Kågebäck, Dubhashi, 2015, Extractive Summarization by Aggregating Multiple Similarities'.
    """

    SENTIMENT_WORDS_PATH = './sublex_1_0.csv'

    def __init__(self, w2v_model, min_sent_len=5, avg_cluster_size=10, a=5.0, beta_=0.5, lambda_=4.0, r_=0.3):
        """
        Initialize the summarizer without actually summarizing anything - use the `summarize` method for that. The
        default values are set to those performing the best in the paper.
        :param w2v_model: trained Word2Vec model
        :param min_sent_len: minimum length of a sentence to be considered for summarization
        :param avg_cluster_size: average number of sentences to put into each cluster
        :param a: >= 0, defines `alpha` = `a` / `n_sentences`; `alpha` defines the threshold between inter-similarity
            of a summary and outer-similarity of a summary to the whole sentence set
        :param beta_: <- [0,1]; `beta` is the coefficient of the convex combination of cluster similarities and query
            similarity, set `beta` = 1 to dismiss the query effect in diversity computation entirely
        :param lambda_: >= 0; `lambda` controls the influence of sentence diversity, the higher `lambda`, the lower
            the summary diversity
        :param r_: >= 0; `r` scales the constraint values of individual sentences, the higher `r`, the higher role
            the sentence constraints play as opposed to the objective function gain
        """
        self.w2v_model = w2v_model
        self.min_sent_len = min_sent_len

        assert a >= 0, 'The parameter `a` must be >= 0'
        assert 0 <= beta_ <= 1, 'The parameter `beta` must be in [0,1]'
        assert lambda_ >= 0, 'The parameter `lambda` must be >= 0'
        assert r_ >= 0, 'The parameter `r` must be >= 0'

        self.a = a
        self.alpha_ = None  # `alpha` = `a` / `n_sentences`
        self.beta_ = beta_  # `beta` = 1 <=> diversity does not take the query into account
        self.lambda_ = lambda_
        self.r_ = r_

        self.n_sentences = None
        self.n_clusters = None  # `n_clusters` = `n_sentences` // `avg_cluster_size`
        self.avg_cluster_size = avg_cluster_size

        self.similarity_matrix = None  # Aggregates pairwise similarities between sentences.
        self.kw_similarities = None  # Vector of sentence-event_keywords similarities.
        self.sentence_cluster_similarities = None  # Pairwise similarities between sentences and clusters.

    def summarize(self, event_keywords, documents, budget, constraint_type):
        """
        Summarize the given event with a given document set under the given constraint type and budget.
        :param event_keywords: keyword representation of the event
        :param documents: document representation of the event, must be instances of the `LemmatizedDocument`
            class defined above
        :param budget: maximum value the constraints can attain altogether
        :param constraint_type: whether to limit the number of words ('words') or the number of sentences ('sentences')
        :return: list of sentences selected for summarization in the same order as they appeared in the given documents
        """
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
        self._cluster_similarities()
        logging.info('Clustered %d sentences into %d clusters in %fs.', self.n_sentences, self.n_clusters, time() - t)

        if constraint_type == 'words':
            constraints = np.fromiter(map(len, sentences_forms), dtype=int, count=self.n_sentences)
        elif constraint_type == 'sentences':
            constraints = np.ones(shape=self.n_sentences, dtype=int)
        else:
            raise ValueError('Invalid value for `mode`. Use either "words" or "sentences".')

        selected_indices = self._greedy_summarization(constraints, budget)
        return [sentences_forms[index] for index in selected_indices]

    def _docs2sents(self, documents):
        """
        Retrieve valid sentences to be considered for summarization. A sentence will be considered if it is at least
        `self.min_sent_len` words long (including punctuation) and contains at least 1 Noun and ` Verb. Additionally,
        retrieve only unique sentences according to their lemmatized representation.
        :param documents: the documents to be converted, must be instances of the `LemmatizedDocument`
            class defined above
        :return: triple (sentences with word forms, sentences with word lemmas, word parts of speech tags) in
            corresponding order
        """

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
        """
        Cluster the sentences into `self.n_clusters` clusters after converting the similarity matrix to a
        distance matrix.
        The clusters will be represented by a matrix `self.sentence_cluster_similarities` mapping sentences to
        their similarities to each cluster. This is useful to quickly compute the current summary diversity.
        """
        distance_matrix = 1.0 - self.similarity_matrix

        # To avoid floating point precision errors.
        np.clip(distance_matrix, 0, 1, out=distance_matrix)
        distance_matrix[np.diag_indices_from(distance_matrix)] = 0.0

        k_medoids = KMedoids(n_clusters=self.n_clusters, distance_metric='precomputed', random_state=1)
        labels = k_medoids.fit_predict(distance_matrix)

        sentence_ids = np.arange(self.n_sentences)
        cluster_adjacency = np.zeros(shape=(self.n_sentences, self.n_clusters), dtype=int)
        cluster_adjacency[sentence_ids, labels] = 1

        self.sentence_cluster_similarities = self.similarity_matrix @ cluster_adjacency

    def _precompute_similarities(self, event_keywords, sentences_lemma, sentences_pos):
        """
        Precompute the matrix aggregating the various sentence similarities. The elements are normalized to [0,1].
        :param event_keywords: keyword representation of the summarized event
        :param sentences_lemma: lemmatized sentences to extract the summary from
        :param sentences_pos: parts of speech tags of the sentences in the same order
        """
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

        # Transform similarities to [0,1].
        min_val = np.min(self.similarity_matrix)
        max_val = np.max(self.similarity_matrix)

        self.similarity_matrix -= min_val
        self.similarity_matrix /= (max_val - min_val)

        # To avoid floating point precision errors.
        self.similarity_matrix[np.diag_indices_from(self.similarity_matrix)] = 1.0

    @staticmethod
    def _sents2vecs(sentences):
        """
        Transform the sentences into TFIDF space.
        :param sentences: lemmatized sentences
        :return: TFIDF matrix
        """
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        return tfidf.fit_transform(sentences)

    def _w2v_similarity(self, sentences):
        """
        Compute Word2Vec similarity of the sentences, defined as cosine similarity between sentence vectors
        obtained by summing together all Word2Vec vectors of the sentence words.
        :param sentences: lemmatized sentences
        :return: matrix of cosine similarities between the vectors
        """
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
        """
        Apply Latent Semantic Indexing to the sentence vectors and compute the resulting vectors similarity.
        :param sentence_vectors: vector representation of the sentences, preferably TFIDF-scaled
        :param k: latent space dimensionality
        :return: matrix cosine similarities between the LSI-transformed vectors
        """
        lsi = TruncatedSVD(n_components=k, random_state=1)
        lsi_matrix = lsi.fit_transform(sentence_vectors)
        return cosine_similarity(lsi_matrix, dense_output=True)

    def _tr_similarity(self, sentences, sentences_pos):
        """
        Compute TextRank similarity of the sentences, defined as Sim(i,j) = |i `intersection` j| / (log|i| + log|j|).
        Only consider Nouns and Verbs for the similarity.
        :param sentences: lemmatized sentences
        :param sentences_pos: parts of speech tags of the sentences in the same order
        :return: matrix of TextRank similarities
        """
        vectorizer = CountVectorizer(binary=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc, dtype=float)

        pos_sentences = [[word for word, pos in zip(sentence, sentence_pos) if pos[0] in 'NV'] for
                         sentence, sentence_pos in zip(sentences, sentences_pos)]

        bow_matrix = vectorizer.fit_transform(pos_sentences)
        log_lengths = np.log(np.fromiter(map(len, sentences), dtype=float, count=self.n_sentences))
        sim = (bow_matrix @ bow_matrix.T) / np.add.outer(log_lengths, log_lengths)

        return sim

    def _kw_similarity(self, sentences, keywords):
        """
        Compute Keyword similarity of the sentences, defined as
        Sim(i,j) = sum_{w <- i `cap` j `cap KW}{TFIDF_w} / (|i| + |j|).
        :param sentences: lemmatized sentences
        :param keywords: keyword representation of the summarized event
        :return: matrix of Keyword similarities
        """
        tfidf = TfidfVectorizer(sublinear_tf=True, analyzer=lambda doc: doc, preprocessor=lambda doc: doc)
        tfidf_matrix = tfidf.fit_transform(sentences)
        kw_indices = [tfidf.vocabulary_[kw] for kw in keywords if kw in tfidf.vocabulary_]
        kw_slice = tfidf_matrix[:, kw_indices]

        self.kw_similarities = np.sum(kw_slice, axis=1)

        kw_slice_copy = kw_slice.copy()
        kw_slice_copy.data.fill(1)
        lengths = np.fromiter(map(len, sentences), dtype=float, count=self.n_sentences)

        return (kw_slice @ kw_slice_copy.T) / np.add.outer(lengths, lengths)

    def _sentiment_similarity(self, sentences, sentiment):
        """
        Compute Sentiment similarity of the sentences, defined as Sim(i,j) = 1 - |score(i) - score(j)|, where
        score(i) is the fraction of words from sentence i appearing in negative (or positive) word list, respectively.
        :param sentences: lemmatized sentences
        :param sentiment: whether to compute 'negative', 'positive' or 'both' sentiment similarities
        :return: matrix of Sentiment similarities
        """
        negative_words, positive_words = self._load_sentiment_words()
        negative_scores = np.empty(self.n_sentences, dtype=float)
        positive_scores = np.empty(self.n_sentences, dtype=float)

        for i, sentence in enumerate(sentences):
            neg_words = sum(1 for _ in filter(lambda word: word in negative_words, sentence))
            pos_words = sum(1 for _ in filter(lambda word: word in positive_words, sentence))

            negative_scores[i] = neg_words / len(sentence)
            positive_scores[i] = pos_words / len(sentence)

        negative_similarity = 1.0 - np.abs(np.subtract.outer(positive_scores, positive_scores))
        positive_similarity = 1.0 - np.abs(np.subtract.outer(negative_scores, negative_scores))

        if sentiment == 'negative':
            return negative_similarity
        elif sentiment == 'positive':
            return positive_similarity

        return np.multiply(negative_similarity, positive_similarity)

    def _load_sentiment_words(self):
        """
        Load the lists of negative and positive words (already lemmatized) from a file.
        :return: negative and positive words in frozen sets
        """
        negative_words = []
        positive_words = []

        with open(self.SENTIMENT_WORDS_PATH, 'r', encoding='utf8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t', fieldnames=['negation', 'pos', 'lemma', 'orientation', 'source'])

            for row in reader:
                lemma = row['lemma'].split('_')[0]  # Strip lemma comment.

                if row['negation'] == 'N':  # We don't have negations in the lemmatized files used for input.
                    lemma = lemma[2:]

                if row['orientation'] == 'POS':
                    positive_words.append(lemma)
                elif row['orientation'] == 'NEG':
                    negative_words.append(lemma)
                else:
                    raise ValueError('Invalid orientation: {:s}'.format(str(row)))

        return frozenset(negative_words), frozenset(positive_words)

    def _greedy_summarization(self, constraints, budget):
        """
        The greedy optimization loop performing the summarization.
        :param constraints: vector of sentence constraints
        :param budget: maximum value the constraints can attain altogether
        :return: indices of the sentences selected for summarization
        """
        summary = []
        remainder = set(range(self.n_sentences))

        cost_so_far = 0
        objective_function = 0
        t = time()

        while len(remainder) > 0:
            k, val = self._feasible_argmax(summary, remainder, objective_function, constraints, cost_so_far, budget)

            if k is not None and val - objective_function >= 0:
                cost_so_far += constraints[k]
                objective_function = val
                summary.append(k)
            else:
                # Can break, since the argmax selected was also feasible. Otherwise, some sentence with a lower
                # objective function gain could still satisfy the constraints.
                break

            remainder.remove(k)

        singleton_candidates = set(filter(lambda singleton: constraints[singleton] <= budget, range(self.n_sentences)))
        fake_constraints = np.ones(self.n_sentences, dtype=int)
        singleton_id, singleton_val = self._feasible_argmax([], singleton_candidates, 0.0, fake_constraints, 0, budget)

        if singleton_val > objective_function:
            logging.info('Summarized event in %fs while filling %d out of %d budget.', time() - t,
                         constraints[singleton_id], budget)
            return [singleton_id]

        logging.info('Summarized event in %fs while filling %d out of %d budget.', time() - t, cost_so_far, budget)
        return summary

    def _feasible_argmax(self, summary, remainder, value_so_far, constraints, cost_so_far, budget):
        """
        Select the sentence index from `remainder` maximizing the objective function gain such that its constraint
        added to `cost_so_far` remains feasible.
        :param summary: indices of sentences selected for the summary so far
        :param remainder: indices of sentences not yet selected for the summarization
        :param value_so_far: objective function value obtained so far
        :param constraints: vector of sentence constraints
        :param cost_so_far: total value of constraints of the sentences selected so far
        :param budget: maximum value the constraints can attain altogether
        :return: tuple (index, objective function value) of the best sentence or (None, 0) if no feasible
            sentence is found
        """
        argmax_id = None
        max_val = -math.inf
        quality = self._quality
        r = self.r_

        for sentence_id in remainder:
            new_objective_value = quality(summary + [sentence_id])
            scaled_constraint = constraints[sentence_id] ** r

            function_gain = (new_objective_value - value_so_far) / scaled_constraint

            if cost_so_far + constraints[sentence_id] <= budget and function_gain > max_val:
                argmax_id = sentence_id
                max_val = function_gain

        if argmax_id is None:
            return None, 0

        return argmax_id, quality(summary + [argmax_id])

    def _quality(self, summary):
        """
        The objective function being optimized. Measures the similarity of the summary to the whole set of sentences
        and its diversity. Denoted `F` in the paper.
        :param summary: indices of sentences selected for the summary so far
        :return: objective function value
        """
        return self._similarity(summary) + self.lambda_ * self._diversity(summary)

    def _similarity(self, summary):
        """
        Similarity of the summary to the whole set of sentences. Denoted `L_1` in the paper.
        :param summary: indices of sentences selected for the summary so far
        :return: similarity
        """
        inter_similarity = np.sum(self.similarity_matrix[:, summary], axis=1)
        outer_similarity = np.sum(self.similarity_matrix, axis=1)

        return np.sum(np.minimum(inter_similarity, self.alpha_ * outer_similarity))

    def _diversity(self, summary):
        """
        Diversity of the summary. Denoted `R_1` in the paper.
        :param summary: indices of sentences selected for the summary so far
        :return: diversity
        """
        summary_reward = np.mean(self.sentence_cluster_similarities[summary], axis=0)

        if self.kw_similarities is None:
            return np.sum(np.sqrt(summary_reward))

        keyword_reward = self.kw_similarities[summary]
        return np.sum(np.sqrt(self.beta_ * summary_reward + (1 - self.beta_) * keyword_reward))
