import collections
import numbers

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin


class OnlineVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=1, max_features=None, binary=False, dtype=np.int64):

        self.input = input
        self.encoding = encoding
        self.decode_error = decode_error
        self.strip_accents = strip_accents
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df

        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")

        self.max_features = max_features

        if max_features is not None:
            if not isinstance(max_features, numbers.Integral) or max_features <= 0:
                raise ValueError("max_features=%r, neither a positive integer nor None" % max_features)

        self.ngram_range = ngram_range
        self.binary = binary
        self.dtype = dtype
        self.bow_matrix = None

        self.vocabulary_ = collections.defaultdict()
        self.vocabulary_.default_factory = self.vocabulary_.__len__

    def _append_matrix(self, batch_matrix):
        if self.bow_matrix is None:
            self.bow_matrix = batch_matrix
            return self.bow_matrix

        # Pad the current BOW matrix by zero columns from the right to compensate for new words in this batch.
        new_width = max(batch_matrix.shape[1], self.bow_matrix.shape[1])
        self.bow_matrix = sp.csr_matrix((self.bow_matrix.data, self.bow_matrix.indices, self.bow_matrix.indptr),
                                        copy=False, shape=(self.bow_matrix.shape[0], new_width))

        self.bow_matrix = sp.vstack((self.bow_matrix, batch_matrix))
        return self.bow_matrix

    def _count_vocab(self, raw_documents):
        analyzer = self.build_analyzer()
        vocabulary = self.vocabulary_

        values = []
        j_indices = []
        indptr = [0]

        for doc in raw_documents:
            token_counter = collections.Counter(vocabulary[token] for token in analyzer(doc))
            j_indices.extend(token_counter.keys())
            values.extend(token_counter.values())
            indptr.append(len(j_indices))

        n_tokens = max(j_indices) + 1

        if n_tokens < 0:
            raise ValueError('No tokens found in the document batch.')

        if self.bow_matrix is not None and n_tokens < self.bow_matrix.shape[1]:
            n_tokens = self.bow_matrix.shape[1]

        values = np.asarray(values, dtype=np.intc)
        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.asarray(indptr, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, n_tokens), dtype=self.dtype)
        X.sort_indices()

        return X

    def _sort_features(self, X):
        """Sort features by name

        Returns a reordered matrix and modifies the vocabulary in place
        """
        sorted_features = sorted(self.vocabulary_.items())
        map_index = np.empty(len(sorted_features), dtype=np.int32)

        for new_val, (term, old_val) in enumerate(sorted_features):
            self.vocabulary_[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        if isinstance(raw_documents, str):
            raise ValueError('Iterable over raw text documents expected, string object received.')

        X = self._count_vocab(raw_documents)

        if self.binary:
            X.data.fill(1)

        return self._sort_features(self._append_matrix(X))

    def partial_fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents):
        pass

    def inverse_transform(self, X):
        pass

    def get_feature_names(self):
        pass


if __name__ == '__main__':
    from sklearn.feature_extraction.text import CountVectorizer
    from event_detection import data_fetchers
    from time import time

    docs, _ = data_fetchers.fetch_czech_corpus(num_docs=10000000)
    subset = docs

    count_vectorizer = CountVectorizer(binary=True)
    online_vectorizer = OnlineVectorizer(binary=True)

    # t0 = time()
    # count_vectorizer.fit_transform(subset)
    # print('Count vectorizer done in %fs.' % (time() - t0))

    t0 = time()
    online_vectorizer.fit_transform(subset)
    print('Online vectorizer done in %fs.' % (time() - t0))
