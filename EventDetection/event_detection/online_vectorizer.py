import numbers
from collections import Counter, defaultdict
from operator import itemgetter

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import VectorizerMixin


class OnlineVectorizer(BaseEstimator, VectorizerMixin):
    def __init__(self, input='content', encoding='utf-8', decode_error='strict', strip_accents=None, lowercase=True,
                 preprocessor=None, tokenizer=None, stop_words=None, token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1),
                 analyzer='word', max_df=1.0, min_df=0, max_features=None, binary=False, dtype=np.int64):

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

        self.bow_matrix_ = None
        self.vocabulary_ = defaultdict()
        self.vocabulary_.default_factory = self.vocabulary_.__len__

    def _append_matrix(self, batch_matrix):
        """
        Append the given term-document matrix from a document batch to the BOW matrix in place.
        :param batch_matrix: sparse matrix of term frequencies from the currently processed document batch
        :return: self.bow_matrix after appending it with batch_matrix
        """
        if self.bow_matrix_ is None:
            self.bow_matrix_ = batch_matrix
            return self.bow_matrix_

        # Pad the current BOW matrix by zero columns from the right to compensate for new words in this batch.
        new_width = max(batch_matrix.shape[1], self.bow_matrix_.shape[1])
        self.bow_matrix_ = sp.csr_matrix((self.bow_matrix_.data, self.bow_matrix_.indices, self.bow_matrix_.indptr),
                                         copy=False, shape=(self.bow_matrix_.shape[0], new_width))

        batch_matrix = sp.csr_matrix((batch_matrix.data, batch_matrix.indices, batch_matrix.indptr),
                                     copy=False, shape=(batch_matrix.shape[0], new_width))

        self.bow_matrix_ = sp.vstack((self.bow_matrix_, batch_matrix))
        return self.bow_matrix_

    def _count_vocab(self, raw_documents):
        """
        Count term frequencies of the given document batch and return them in a sparse CSR matrix format.
        :param raw_documents: list of documents as strings
        :return: CSR sparse matrix of shape (len(raw_documents), len(self.vocabulary_))
        """
        analyzer = self.build_analyzer()
        vocabulary = self.vocabulary_

        values = []
        j_indices = []
        indptr = [0]

        for doc in raw_documents:
            token_counter = Counter(vocabulary[token] for token in analyzer(doc))
            j_indices.extend(token_counter.keys())
            values.extend(token_counter.values())
            indptr.append(len(j_indices))

        values = np.asarray(values, dtype=np.intc)
        j_indices = np.asarray(j_indices, dtype=np.intc)
        indptr = np.asarray(indptr, dtype=np.intc)

        X = sp.csr_matrix((values, j_indices, indptr), shape=(len(indptr) - 1, len(vocabulary)), dtype=self.dtype)
        X.sort_indices()

        return X

    def _sort_features(self, X):
        """
        Sort features by name and modify the vocabulary in place.
        :param X:
        :return: a reordered matrix X
        """
        vocabulary = self.vocabulary_
        sorted_features = sorted(vocabulary.items())
        map_index = np.empty(len(sorted_features), dtype=np.int32)

        for new_val, (term, old_val) in enumerate(sorted_features):
            vocabulary[term] = new_val
            map_index[old_val] = new_val

        X.indices = map_index.take(X.indices, mode='clip')
        return X

    def limit_features(self, X, min_df=None, max_df=None, max_features=None):
        min_df = min_df or self.min_df
        max_df = max_df or self.max_df
        max_features = max_features or self.max_features

        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")

        if max_features is not None:
            if not isinstance(max_features, numbers.Integral) or max_features <= 0:
                raise ValueError("max_features=%r, neither a positive integer nor None" % max_features)

        n_docs = X.shape[0]
        high = max_df if isinstance(max_df, numbers.Integral) else max_df * n_docs
        low = min_df if isinstance(min_df, numbers.Integral) else min_df * n_docs

        if high < low:
            raise ValueError('max_df corresponds to < documents than min_df')

        dfs = X.getnnz(axis=0)  # Document frequency.
        tfs = np.asarray(X.sum(axis=0)).ravel()  # Term frequency.
        mask = np.ones(len(dfs), dtype=bool)

        if high is not None:
            mask &= dfs <= high

        if low is not None:
            mask &= dfs >= low

        if max_features is not None and mask.sum() > max_features:
            mask_inds = (-tfs[mask]).argsort()[:max_features]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1
        vocabulary = self.vocabulary_

        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]

        kept_indices = np.where(mask)[0]

        if len(kept_indices) == 0:
            raise ValueError('After pruning, no terms remain. Try a lower min_df or a higher max_df.')

        return X[:, kept_indices]

    def fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def fit_transform(self, raw_documents, y=None):
        if isinstance(raw_documents, str):
            raise ValueError('Iterable over raw text documents expected, string object received.')

        X = self._count_vocab(raw_documents)

        if self.binary:
            X.data.fill(1)

        X = self.limit_features(X)
        self._append_matrix(X)

        return self._sort_features(self.bow_matrix_)

    def partial_fit(self, raw_documents, y=None):
        self.fit_transform(raw_documents, y)
        return self

    def transform(self, raw_documents):
        pass

    def inverse_transform(self, X):
        pass

    def get_feature_names(self):
        """
        Array mapping from feature integer indices to feature name.
        """
        return [t for t, _ in sorted(self.vocabulary_.items(), key=itemgetter(1))]


if __name__ == '__main__':
    from event_detection import data_fetchers
    from time import time

    # docs, _ = data_fetchers.fetch_czech_corpus(num_docs=10000000)
    docs, _ = data_fetchers.fetch_czech_corpus_dec_jan()
    subset = docs

    step_size = 164234
    j = step_size
    online_vectorizer = OnlineVectorizer(binary=True, min_df=30, max_df=0.9)

    t0 = time()

    for i in range(0, len(subset), step_size):
        online_vectorizer.fit_transform(subset[i:j])
        print(online_vectorizer.bow_matrix_.shape)
        j += step_size

    print('Done in %fs' % (time() - t0))
