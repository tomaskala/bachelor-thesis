import unittest

import numpy.testing as npt
from sklearn.feature_extraction.text import CountVectorizer

from event_detection.online_vectorizer import OnlineVectorizer

SMALL_SET_1 = ['This is my first document', 'This is my second document', 'This is my first test set']
SMALL_SET_2 = ['Really random sentence', 'This is my third document']
SMALL_SET_3 = ['Some totally random text', 'This sentence will be a little bit longer than usual hopefully it works']
SMALL_SET_4 = ['Turtles are pretty', 'Why so serious', 'Let us have one more sentence just for fun']

DOCS_1 = ['Common words suck', 'However, rare words tend to be penalized too, common!', 'Common words penalized']
DOCS_2 = ['These common words are everywhere!', 'Where do they come from?', 'Common words are common']

COUNTED_SET = ['one two three four five six seven eight nine ten', 'two three four five six seven eight nine ten',
               'three four five six seven eight nine ten', 'four five six seven eight nine ten',
               'five six seven eight nine ten', 'six seven eight nine ten', 'seven eight nine ten', 'eight nine ten',
               'nine ten', 'ten']


class OnlineVectorizerLimitTest(unittest.TestCase):
    def setUp(self):
        self.online_vectorizer = OnlineVectorizer(binary=True)
        self.reference_vectorizer = CountVectorizer(binary=True)

    def assert_sparse_csr_equal(self, x, y, err_msg):
        x = x.sorted_indices()
        y = y.sorted_indices()
        self.assertEqual(x.shape, y.shape, err_msg + ', shape')
        self.assertEqual(x.dtype, y.dtype, err_msg + ', dtype')
        self.assertEqual(x.getnnz(), y.getnnz(), err_msg + ', nnz')
        npt.assert_array_equal(x.data, y.data, err_msg + ', data')
        npt.assert_array_equal(x.indices, y.indices, err_msg + ', indices')
        npt.assert_array_equal(x.indptr, y.indptr, err_msg + ', indptr')

    # def test_automatic_calls(self):
    #     self.online_vectorizer.trim_every_n_docs = 2
    #
    #     with mock.patch.object(self.online_vectorizer, 'limit_features') as mocked:
    #         call_count = mocked.call_count
    #
    #         self.online_vectorizer.fit_transform(SMALL_SET_1[:2])  # Fit 2 documents.
    #         mocked.assert_called()
    #         self.assertEqual(self.online_vectorizer.accumulated_documents, 0)
    #         call_count += mocked.call_count
    #
    #         mocked.reset_mock()
    #
    #         self.online_vectorizer.fit_transform(SMALL_SET_2)  # Fit 2 documents.
    #         mocked.assert_called()
    #         self.assertEqual(self.online_vectorizer.accumulated_documents, 0)
    #         call_count += mocked.call_count
    #
    #         mocked.reset_mock()
    #
    #         self.online_vectorizer.fit_transform(SMALL_SET_3)  # Fit 2 documents.
    #         mocked.assert_called()
    #         self.assertEqual(self.online_vectorizer.accumulated_documents, 0)
    #         call_count += mocked.call_count
    #
    #         mocked.reset_mock()
    #
    #         self.online_vectorizer.fit_transform([SMALL_SET_1[2]])  # Fit 1 document.
    #         mocked.assert_not_called()
    #         self.assertEqual(self.online_vectorizer.accumulated_documents, 1)
    #         call_count += mocked.call_count
    #
    #         mocked.reset_mock()
    #
    #         self.online_vectorizer.fit_transform(SMALL_SET_4)  # Fit 3 documents.
    #         mocked.assert_called()
    #         self.assertEqual(self.online_vectorizer.accumulated_documents, 0)
    #         call_count += mocked.call_count
    #
    #         self.assertEqual(call_count, 4)

    def test_offline_limiting_min(self):
        self.online_vectorizer.min_df = 3
        self.reference_vectorizer.min_df = 3

        self.online_vectorizer.fit_transform(DOCS_1 + DOCS_2)
        online_bow = self.online_vectorizer.limit_features(self.online_vectorizer.bow_matrix_)

        reference_bow = self.reference_vectorizer.fit_transform(DOCS_1 + DOCS_2)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Offline features limiting, min, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Offline features limiting, min, dict')

    def test_offline_limiting_max(self):
        self.online_vectorizer.max_df = 2
        self.reference_vectorizer.max_df = 2

        self.online_vectorizer.fit_transform(DOCS_1 + DOCS_2)
        online_bow = self.online_vectorizer.limit_features(self.online_vectorizer.bow_matrix_)

        reference_bow = self.reference_vectorizer.fit_transform(DOCS_1 + DOCS_2)

        self.assert_sparse_csr_equal(online_bow, reference_bow, 'Offline features limiting, max, bow')
        self.assertDictEqual(self.online_vectorizer.vocabulary_, self.reference_vectorizer.vocabulary_,
                             'Offline features limiting, max, dict')

    def test_online_limiting_max(self):
        self.online_vectorizer.max_df = 2

        sets = [{'two', 'three'},
                {'five', 'six'},
                {'eight', 'nine'}]

        for i in range(len(COUNTED_SET) - 1, 1, -3):
            doc1 = COUNTED_SET[i]
            doc2 = COUNTED_SET[i - 1]
            doc3 = COUNTED_SET[i - 2]

            self.online_vectorizer.fit_transform([doc1, doc2, doc3])
            self.assertEqual(sets[i // 3 - 1], set(self.online_vectorizer.vocabulary_.keys()),
                             'Online features limiting, dict')


if __name__ == '__main__':
    unittest.main()
