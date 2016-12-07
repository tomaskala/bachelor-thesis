import unittest

import numpy.testing as npt
from sklearn.feature_extraction.text import CountVectorizer

from event_detection import data_fetchers
from event_detection.online_vectorizer import OnlineVectorizer

SMALL_SET_1 = ['This is my first document', 'This is my second document', 'This is my first test set']
SMALL_SET_2 = ['Really random sentence', 'This is my third document']
SMALL_SET_3 = ['Some totally random text', 'This sentence will be a little bit longer than usual hopefully it works']
SMALL_SET_4 = ['Turtles are pretty', 'Why so serious', 'Let us have one more sentence just for fun']

LARGER = ['Such words much wow']
SMALLER = ['Such much']

OLD_WORDS = ['This document contains some words']
NEW_WORDS = ['None of these was seen before']


class OnlineVectorizerBasicTest(unittest.TestCase):
    def setUp(self):
        self.online_vectorizer = OnlineVectorizer(binary=True)
        self.reference_vectorizer = CountVectorizer(binary=True)

    @classmethod
    def setUpClass(cls):
        cls.docs, _ = data_fetchers.fetch_czech_corpus(num_docs=1000)

    def assert_sparse_csr_equal(self, x, y, err_msg):
        x = x.sorted_indices()
        y = y.sorted_indices()
        self.assertEqual(x.shape, y.shape, err_msg + ', shape')
        self.assertEqual(x.dtype, y.dtype, err_msg + ', dtype')
        self.assertEqual(x.getnnz(), y.getnnz(), err_msg + ', nnz')
        npt.assert_array_equal(x.data, y.data, err_msg + ', data')
        npt.assert_array_equal(x.indices, y.indices, err_msg + ', indices')
        npt.assert_array_equal(x.indptr, y.indptr, err_msg + ', indptr')

    def test_one_docset(self):
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_1)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Offline fit, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Offline fit, dict')

    def test_two_docsets_online(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_2)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_2)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, two docsets, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, two docsets, dict')

    def test_two_fits_single_docset(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_1)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_1)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Fit same docset twice, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Fit same docset twice, dict')

    def test_more_partial_fits(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        self.online_vectorizer.fit_transform(SMALL_SET_2)
        self.online_vectorizer.fit_transform(SMALL_SET_3)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_4)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_2 + SMALL_SET_3 + SMALL_SET_4)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, four docsets, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, four docsets, dict')

    def test_large_docset_offline(self):
        subset = self.docs

        self.online_vectorizer.fit_transform(subset[:200])
        self.online_vectorizer.fit_transform(subset[200:400])
        self.online_vectorizer.fit_transform(subset[400:600])
        self.online_vectorizer.fit_transform(subset[600:800])
        online_bow = self.online_vectorizer.fit_transform(subset[800:])

        reference_bow = self.reference_vectorizer.fit_transform(subset)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, large docset, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, large docset, dict')

    def test_fill_from_left_in_batch(self):
        # The first fit creates the BOW matrix. If the second batch contains only words from the first, say, half
        # of the BOW matrix and no new words, index error will occur.
        # Ex:
        # First: ['Such words much wow']
        # [[1, 1, 1, 1]]
        #
        # Second: ['Such much']
        # [[1, 1, 1, 1]
        #  [1, 1, 0, 0]]
        # max(j_indices) + 1 = 2 -> cols(bow_matrix) = 4, cols(batch_matrix) = 2 -> cannot vstack.

        self.online_vectorizer.fit_transform(LARGER)
        online_bow = self.online_vectorizer.fit_transform(SMALLER)

        reference_bow = self.reference_vectorizer.fit_transform(LARGER + SMALLER)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, fill from left, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, fill from left, dict')

    def test_two_documents(self):
        docs = [SMALL_SET_1[0], SMALL_SET_1[1]]

        self.online_vectorizer.fit_transform([docs[0]])
        online_bow = self.online_vectorizer.fit_transform([docs[1]])

        reference_bow = self.reference_vectorizer.fit_transform(docs)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, two documents, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, two documents, dict')

    def test_large_docset_small_batches(self):
        subset = self.docs

        for doc in subset[:-1]:
            self.online_vectorizer.fit_transform([doc])

        online_bow = self.online_vectorizer.fit_transform([subset[-1]])

        reference_bow = self.reference_vectorizer.fit_transform(subset)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, large docset with small batches, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, large docset with small batches, dict')

    def test_new_words_only(self):
        self.online_vectorizer.fit_transform(OLD_WORDS)
        online_bow = self.online_vectorizer.fit_transform(NEW_WORDS)

        reference_bow = self.reference_vectorizer.fit_transform(OLD_WORDS + NEW_WORDS)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Online fit, new words only, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, new words only, dict')

    def test_commutativity(self):
        self.online_vectorizer.fit_transform(OLD_WORDS)
        online_bow1 = self.online_vectorizer.fit_transform(NEW_WORDS)

        online_vectorizer2 = OnlineVectorizer(binary=True)
        online_vectorizer2.fit_transform(NEW_WORDS)
        online_bow2 = online_vectorizer2.fit_transform(OLD_WORDS)

        self.assert_sparse_csr_equal(online_bow1, online_bow2[::-1], 'Commutativity, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, online_vectorizer2.vocabulary_, 'Commutativity, dict')


if __name__ == '__main__':
    unittest.main()
