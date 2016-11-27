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


class TestOnlineVectorizer(unittest.TestCase):
    def setUp(self):
        self.online_vectorizer = OnlineVectorizer(binary=True)
        self.reference_vectorizer = CountVectorizer(binary=True)

    @classmethod
    def setUpClass(cls):
        cls.docs, _ = data_fetchers.fetch_czech_corpus_dec_jan()

    def test_one_docset(self):
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_1)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Offline fit, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Offline fit, dicts')

    def test_two_docsets_online(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_2)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_2)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Online fit, two docsets, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, two docsets, dicts')

    def test_two_fits_single_docset(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_1)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_1)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Fit same docset twice, dict')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Fit same docset twice, bow')

    def test_more_partial_fits(self):
        self.online_vectorizer.fit_transform(SMALL_SET_1)
        self.online_vectorizer.fit_transform(SMALL_SET_2)
        self.online_vectorizer.fit_transform(SMALL_SET_3)
        online_bow = self.online_vectorizer.fit_transform(SMALL_SET_4)

        reference_bow = self.reference_vectorizer.fit_transform(SMALL_SET_1 + SMALL_SET_2 + SMALL_SET_3 + SMALL_SET_4)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Online fit, four docsets, dict')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, four docsets, bow')

    def test_large_dataset_offline(self):
        subset = self.docs[:1000]

        self.online_vectorizer.fit_transform(subset[:200])
        self.online_vectorizer.fit_transform(subset[200:400])
        self.online_vectorizer.fit_transform(subset[400:600])
        self.online_vectorizer.fit_transform(subset[600:800])
        online_bow = self.online_vectorizer.fit_transform(subset[800:])

        reference_bow = self.reference_vectorizer.fit_transform(subset)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Online fit, large docset, dict')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, large docset, bow')

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

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Online fit, fill from left, dict')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, fill from left, bow')

    def test_two_documents(self):
        docs = [SMALL_SET_1[0], SMALL_SET_1[1]]

        self.online_vectorizer.fit_transform([docs[0]])
        online_bow = self.online_vectorizer.fit_transform([docs[1]])

        reference_bow = self.reference_vectorizer.fit_transform(docs)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(), 'Online fit, two documents, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, two documents, dict')

    def test_large_dataset_small_batches(self):
        subset = self.docs[:1000]

        for doc in subset[:-1]:
            self.online_vectorizer.fit_transform([doc])

        online_bow = self.online_vectorizer.fit_transform([subset[-1]])

        reference_bow = self.reference_vectorizer.fit_transform(subset)

        npt.assert_array_equal(online_bow.toarray(), reference_bow.toarray(),
                               'Online fit, large docset with small strides, dict')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Online fit, large docset with small strides, bow')


if __name__ == '__main__':
    unittest.main()
