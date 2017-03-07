import os

import numpy as np

CZECH_LEMMATIZED_TEXTS = '../../lemmatized_docs'
CZECH_LEMMATIZED_TEXTS_DEC_JAN = ['2014-12-01.txt.tagged', '2014-12-02.txt.tagged', '2014-12-03.txt.tagged',
                                  '2014-12-04.txt.tagged', '2014-12-05.txt.tagged', '2014-12-06.txt.tagged',
                                  '2014-12-07.txt.tagged', '2014-12-08.txt.tagged', '2014-12-09.txt.tagged',
                                  '2014-12-10.txt.tagged', '2014-12-11.txt.tagged', '2014-12-12.txt.tagged',
                                  '2014-12-13.txt.tagged', '2014-12-14.txt.tagged', '2014-12-15.txt.tagged',
                                  '2014-12-16.txt.tagged', '2014-12-17.txt.tagged', '2014-12-18.txt.tagged',
                                  '2014-12-19.txt.tagged', '2014-12-20.txt.tagged', '2014-12-21.txt.tagged',
                                  '2014-12-22.txt.tagged', '2014-12-23.txt.tagged', '2014-12-24.txt.tagged',
                                  '2014-12-25.txt.tagged', '2014-12-26.txt.tagged', '2014-12-27.txt.tagged',
                                  '2014-12-28.txt.tagged', '2014-12-29.txt.tagged', '2014-12-30.txt.tagged',
                                  '2014-12-31.txt.tagged', '2015-01-01.txt.tagged', '2015-01-02.txt.tagged',
                                  '2015-01-03.txt.tagged', '2015-01-04.txt.tagged', '2015-01-05.txt.tagged',
                                  '2015-01-06.txt.tagged', '2015-01-07.txt.tagged', '2015-01-08.txt.tagged',
                                  '2015-01-09.txt.tagged', '2015-01-10.txt.tagged', '2015-01-11.txt.tagged',
                                  '2015-01-12.txt.tagged', '2015-01-13.txt.tagged', '2015-01-14.txt.tagged',
                                  '2015-01-15.txt.tagged', '2015-01-16.txt.tagged', '2015-01-17.txt.tagged',
                                  '2015-01-18.txt.tagged', '2015-01-19.txt.tagged', '2015-01-20.txt.tagged',
                                  '2015-01-21.txt.tagged', '2015-01-22.txt.tagged', '2015-01-23.txt.tagged',
                                  '2015-01-24.txt.tagged', '2015-01-25.txt.tagged', '2015-01-26.txt.tagged',
                                  '2015-01-27.txt.tagged', '2015-01-28.txt.tagged', '2015-01-29.txt.tagged',
                                  '2015-01-30.txt.tagged', '2015-01-31.txt.tagged']

# Total: 2,078,774
DOCS_IN_DAYS = np.array(
    [2985, 5413, 5529, 3042, 3144, 6056, 5980, 6250, 5887, 5883, 3203, 3284, 6105, 5945, 6258, 6071, 5789, 3444, 3184,
     6094, 5986, 6293, 6134, 3835, 4818, 3249, 6217, 5956, 6308, 6002, 6162, 3325, 3388, 6223, 6246, 6168, 6181, 5766,
     3340, 3086, 6190, 6060, 6218, 6314, 5941, 3046, 3123, 6041, 6190, 6205, 6184, 5770, 3546, 3327, 6297, 6086, 6294,
     6071, 6291, 3189, 3617, 6349, 6483, 6380, 6299, 5788, 3407, 3423, 6161, 6261, 6342, 6141, 5979, 3413, 3511, 6238,
     6349, 6549, 6095, 6086, 3434, 3418, 6218, 6192, 6228, 6318, 5975, 3309, 2957, 6385, 6399, 6525, 6471, 6045, 3291,
     3410, 6251, 6300, 6312, 6585, 5955, 3374, 3279, 6561, 6308, 6299, 6191, 5340, 3125, 2914, 3220, 6202, 6222, 6239,
     6187, 3657, 3607, 6575, 6527, 6376, 4068, 5573, 3520, 3475, 6400, 6177, 6496, 3726, 5553, 3459, 3550, 6657, 6438,
     6548, 6423, 5816, 3481, 3480, 6189, 6083, 6045, 6186, 5778, 3559, 3498, 6459, 6095, 6272, 5975, 5819, 3445, 3529,
     6265, 6253, 6022, 6133, 5762, 3068, 3091, 6177, 6249, 6042, 6108, 5725, 3403, 3296, 6009, 6158, 5795, 5920, 5801,
     3220, 2981, 5938, 5866, 6173, 6084, 5642, 3145, 2953, 5874, 5911, 5770, 5643, 5284, 3246, 2968, 5469, 3067, 7596,
     5619, 5488, 3202, 3099, 5586, 5897, 5524, 5658, 5472, 2998, 2922, 5927, 5659, 5719, 5983, 5424, 3246, 3044, 5753,
     6178, 5872, 5874, 5262, 3080, 3011, 5659, 5786, 5686, 5724, 5345, 2982, 2943, 5634, 5713, 5681, 5725, 5396, 3286,
     3102, 5636, 5706, 5462, 5783, 5217, 3210, 3264, 5606, 5826, 5751, 5889, 5618, 3270, 3326, 6264, 6072, 6158, 6399,
     5804, 3244, 3360, 6307, 5990, 6204, 5979, 5596, 3314, 3347, 6097, 6079, 5978, 6251, 6147, 3391, 3412, 6126, 6171,
     6315, 6077, 6162, 3537, 3420, 6239, 6332, 6731, 6650, 6514, 3358, 3469, 6734, 6710, 6620, 6715, 6458, 3960, 3696,
     6712, 6480, 6559, 6488, 6361, 3728, 3168, 6818, 6319, 6396, 3389, 8634, 3311, 3274, 5707, 3936, 6244, 6586, 6292,
     3506, 3448, 6404, 6346, 6405, 6371, 6087, 3544, 3371, 6517, 6221, 6667, 6185, 6228, 3386, 3164, 3802, 6386, 6621,
     6530, 6030, 3406, 3445, 6521, 6614, 6416, 6464, 6077, 3444, 3548, 7325, 6553, 6743, 6603, 6060, 3521, 3385, 6369,
     6315, 6549, 6501, 6129, 3272, 3316, 6492, 6190, 6509, 6835, 6028, 3256, 3092, 5550, 4796, 2688, 2677, 2862, 2904,
     2928, 5022, 4722, 3863, 3106, 4794, 3272, 3276, 6513, 6598, 6561, 6563, 6372, 3841, 3735, 6645, 6549, 6411, 6462,
     6158, 3396, 3438, 6586, 6576, 6587, 6830, 6505, 3686, 3677, 6829, 6547, 6797, 6726, 6616, 3858])

# Total: 326,565
DOCS_IN_DAYS_DEC_JAN = np.array(
    [7325, 6553, 6743, 6603, 6060, 3521, 3385, 6369, 6315, 6549, 6501, 6129, 3272, 3316, 6492, 6190, 6509, 6835, 6028,
     3256, 3092, 5550, 4796, 2688, 2677, 2862, 2904, 2928, 5022, 4722, 3863, 3106, 4794, 3272, 3276, 6513, 6598, 6561,
     6563, 6372, 3841, 3735, 6645, 6549, 6411, 6462, 6158, 3396, 3438, 6586, 6576, 6587, 6830, 6505, 3686, 3677, 6829,
     6547, 6797, 6726, 6616, 3858])


class LemmatizedDocument:
    """
    Used for everything except summarization.
    """
    __slots__ = ['name', 'text']

    def __init__(self):
        self.name = []
        self.text = []


class CzechLemmatizedTexts:
    """
    By default, extract all POS except for Unknown (X) and Punctuation (Z). POS tags:
    A ... Adjective
    C ... Numeral
    D ... Adverb
    I ... Interjection
    J ... Conjunction
    N ... Noun
    P ... Pronoun
    V ... Verb
    R ... Preposition
    T ... Particle
    X ... Unknown, Not Determined, Unclassifiable
    Z ... Punctuation (also used for the Sentence Boundary token)
    """

    def __init__(self, dataset, fetch_forms, pos=('A', 'C', 'D', 'I', 'J', 'N', 'P', 'V', 'R', 'T'), names_only=False):
        """
        Initialize the document fetcher which yields `LemmatizedDocument` objects upon iteration.
        :param dataset: which dataset to use (full/dec-jan)
        :param fetch_forms: whether to fetch forms or lemmas of the individual words
        :param pos: which POS categories to fetch
        :param names_only: whether to fetch only document names (to save memory) or whole documents
        """
        if dataset == 'full':
            self.document_paths = os.listdir(CZECH_LEMMATIZED_TEXTS)
            self.relative_days = DOCS_IN_DAYS
        elif dataset == 'dec-jan':
            self.document_paths = CZECH_LEMMATIZED_TEXTS_DEC_JAN
            self.relative_days = DOCS_IN_DAYS_DEC_JAN
        else:
            raise ValueError('Unknown dataset - select either `full` or `dec-jan`.')

        self.fetch_forms = fetch_forms
        self.pos = frozenset(pos)
        self.names_only = names_only

    def __iter__(self):
        in_title = False
        in_text = False

        fetch_forms = self.fetch_forms
        pos = self.pos
        names_only = self.names_only

        for file in self.document_paths:
            with open(os.path.join(CZECH_LEMMATIZED_TEXTS, file), 'r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        # Control lines
                        if line == '#docstart':
                            document = LemmatizedDocument()
                        elif line == '#titlestart':
                            in_title = True
                        elif line == '#titleend':
                            in_title = False
                        elif line == '#textstart':
                            in_text = True
                        elif line == '#textend':
                            in_text = False
                        elif line == '#docend':
                            yield document
                        else:
                            # Something else, though ignore the '#' sign by itself, as it appears in some documents.
                            if line.split('\t')[0] == '#':
                                continue

                            raise ValueError('Invalid control line encountered: {}'.format(line))
                    elif not line:
                        # Skip empty lines.
                        continue
                    else:
                        # Actual content
                        split = line.split('\t')

                        if len(split) != 3:
                            continue

                        # Skip undesired POS tags.
                        if split[2][0] not in pos:
                            continue

                        if in_title:
                            if fetch_forms:
                                document.name.append(split[0].strip())
                            else:
                                document.name.append(split[1].strip())
                        elif in_text and not names_only:
                            if fetch_forms:
                                document.text.append(split[0].strip())
                            else:
                                document.text.append(split[1].strip())

    def fetch_relative_days(self):
        """
        Return a numpy array of relative days for the dataset. Implemented separately so the original texts can be
        comfortably streamed while keeping relative days in an array.
        :return: numpy array of relative days in the same order as the documents
        """
        return np.repeat(np.arange(self.relative_days.size), self.relative_days)


class SummarizationDocument:
    """
    Used for summarization only.
    """
    __slots__ = ['name_forms', 'name_lemma', 'sentences_forms', 'sentences_lemma', 'name_pos', 'sentences_pos']

    def __init__(self):
        self.name_forms = []
        self.name_lemma = []
        self.sentences_forms = []
        self.sentences_lemma = []
        self.name_pos = []
        self.sentences_pos = []


class CzechSummarizationTexts:
    def __init__(self, dataset):
        """
        Initialize the document fetcher which yields `SummarizationDocument` objects upon iteration. This fetcher is
        used when summarizing the events. The documents have headlines, text and parts of speech for forms as well
        as lemmas, so they take up more memory. Furthermore, the text is not a list of words as in `LemmatizedDocument`
        objects, but divided into sentences (each sentence being a list of words).

        Intended to load a small sample of documents once it is known they represent an event.
        :param dataset: which dataset to use (full/dec-jan)
        """
        if dataset == 'full':
            self.document_paths = os.listdir(CZECH_LEMMATIZED_TEXTS)
        elif dataset == 'dec-jan':
            self.document_paths = CZECH_LEMMATIZED_TEXTS_DEC_JAN
        else:
            raise ValueError('Unknown dataset - select either `full` or `dec-jan`.')

    def __iter__(self):
        in_title = False
        in_text = False
        sentence_forms = []
        sentence_lemma = []
        sentence_pos = []

        for file in self.document_paths:
            with open(os.path.join(CZECH_LEMMATIZED_TEXTS, file), 'r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        # Control lines
                        if line == '#docstart':
                            document = SummarizationDocument()
                        elif line == '#titlestart':
                            in_title = True
                        elif line == '#titleend':
                            in_title = False
                        elif line == '#textstart':
                            in_text = True
                        elif line == '#textend':
                            in_text = False

                            # End of the last document sentence.
                            if len(sentence_forms) > 0 and len(sentence_lemma) > 0 and len(sentence_pos) > 0:
                                document.sentences_forms.append(sentence_forms)
                                document.sentences_lemma.append(sentence_lemma)
                                document.sentences_pos.append(sentence_pos)

                            sentence_forms = []
                            sentence_lemma = []
                            sentence_pos = []
                        elif line == '#docend':
                            yield document
                        else:
                            # Something else, though ignore the '#' sign by itself, as it appears in some documents.
                            if line.split('\t')[0] == '#':
                                continue

                            raise ValueError('Invalid control line encountered: {}'.format(line))
                    elif not line:
                        # Empty lines mark sentence boundaries.
                        if len(sentence_forms) > 0 and len(sentence_lemma) > 0 and len(sentence_pos) > 0:
                            document.sentences_forms.append(sentence_forms)
                            document.sentences_lemma.append(sentence_lemma)
                            document.sentences_pos.append(sentence_pos)

                        sentence_forms = []
                        sentence_lemma = []
                        sentence_pos = []
                    else:
                        # Actual content
                        split = line.split('\t')

                        if len(split) != 3:
                            continue

                        if in_title:
                            document.name_forms.append(split[0].strip())
                            document.name_lemma.append(split[1].strip())
                            document.name_pos.append(split[2].strip())
                        elif in_text:
                            sentence_forms.append(split[0].strip())
                            sentence_lemma.append(split[1].strip())
                            sentence_pos.append(split[2].strip())
