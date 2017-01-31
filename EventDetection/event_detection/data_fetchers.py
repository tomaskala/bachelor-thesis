import logging
import os

import numpy as np

CZECH_DEC_JAN_DATASET = '../../out2DecJan.txt'
CZECH_DATASET = '../../out2.txt'
CZECH_FULL_TEXTS = '../../fulltexts'
CZECH_FULL_TEXTS_DEC_JAN = ['PDFull.2014-12-01.txt', 'PDFull.2014-12-02.txt', 'PDFull.2014-12-03.txt',
                            'PDFull.2014-12-04.txt', 'PDFull.2014-12-05.txt', 'PDFull.2014-12-06.txt',
                            'PDFull.2014-12-07.txt', 'PDFull.2014-12-08.txt', 'PDFull.2014-12-09.txt',
                            'PDFull.2014-12-10.txt', 'PDFull.2014-12-11.txt', 'PDFull.2014-12-12.txt',
                            'PDFull.2014-12-13.txt', 'PDFull.2014-12-14.txt', 'PDFull.2014-12-15.txt',
                            'PDFull.2014-12-16.txt', 'PDFull.2014-12-17.txt', 'PDFull.2014-12-18.txt',
                            'PDFull.2014-12-19.txt', 'PDFull.2014-12-20.txt', 'PDFull.2014-12-21.txt',
                            'PDFull.2014-12-22.txt', 'PDFull.2014-12-23.txt', 'PDFull.2014-12-24.txt',
                            'PDFull.2014-12-25.txt', 'PDFull.2014-12-26.txt', 'PDFull.2014-12-27.txt',
                            'PDFull.2014-12-28.txt', 'PDFull.2014-12-29.txt', 'PDFull.2014-12-30.txt',
                            'PDFull.2014-12-31.txt', 'PDFull.2015-01-01.txt', 'PDFull.2015-01-02.txt',
                            'PDFull.2015-01-03.txt', 'PDFull.2015-01-04.txt', 'PDFull.2015-01-05.txt',
                            'PDFull.2015-01-06.txt', 'PDFull.2015-01-07.txt', 'PDFull.2015-01-08.txt',
                            'PDFull.2015-01-09.txt', 'PDFull.2015-01-10.txt', 'PDFull.2015-01-11.txt',
                            'PDFull.2015-01-12.txt', 'PDFull.2015-01-13.txt', 'PDFull.2015-01-14.txt',
                            'PDFull.2015-01-15.txt', 'PDFull.2015-01-16.txt', 'PDFull.2015-01-17.txt',
                            'PDFull.2015-01-18.txt', 'PDFull.2015-01-19.txt', 'PDFull.2015-01-20.txt',
                            'PDFull.2015-01-21.txt', 'PDFull.2015-01-22.txt', 'PDFull.2015-01-23.txt',
                            'PDFull.2015-01-24.txt', 'PDFull.2015-01-25.txt', 'PDFull.2015-01-26.txt',
                            'PDFull.2015-01-27.txt', 'PDFull.2015-01-28.txt', 'PDFull.2015-01-29.txt',
                            'PDFull.2015-01-30.txt', 'PDFull.2015-01-31.txt']

CZECH_TAGGED_TEXTS = '../../tagged_docs'
CZECH_TAGGED_TEXTS_DEC_JAN = ['2014-12-01.txt.tagged', '2014-12-02.txt.tagged', '2014-12-03.txt.tagged',
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

# Total: 2,090,635
DOCS_IN_MONTHS = [3002, 5442, 5559, 3059, 3163, 6089, 6013, 6284, 5918, 5916, 3221, 3304, 6139, 5978, 6293, 6103, 5821,
                  3463, 3204, 6128, 6018, 6328, 6168, 3855, 4847, 3269, 6252, 5989, 6343, 6036, 6197, 3343, 3408, 6258,
                  6279, 6202, 6215, 5798, 3359, 3105, 6224, 6093, 6253, 6349, 5975, 3064, 3141, 6076, 6227, 6242, 6218,
                  5804, 3568, 3347, 6333, 6121, 6330, 6105, 6325, 3208, 3639, 6385, 6519, 6415, 6335, 5821, 3426, 3443,
                  6195, 6296, 6378, 6174, 6011, 3433, 3532, 6273, 6384, 6586, 6127, 6119, 3455, 3438, 6253, 6226, 6263,
                  6353, 6008, 3328, 2974, 6422, 6433, 6560, 6508, 6079, 3310, 3431, 6286, 6334, 6347, 6622, 5988, 3394,
                  3299, 6598, 6345, 6336, 6226, 5369, 3143, 2932, 3239, 6237, 6257, 6273, 6226, 3682, 3633, 6615, 6567,
                  6414, 4094, 5606, 3542, 3498, 6439, 6212, 6532, 3746, 5586, 3480, 3573, 6698, 6478, 6588, 6460, 5848,
                  3501, 3500, 6223, 6117, 6078, 6221, 5811, 3580, 3519, 6497, 6129, 6309, 6007, 5854, 3467, 3552, 6302,
                  6291, 6056, 6167, 5794, 3086, 3110, 6212, 6283, 6076, 6144, 5758, 3424, 3317, 6046, 6195, 5829, 5952,
                  5833, 3236, 2997, 5972, 5900, 6206, 6119, 5674, 3163, 2971, 5907, 5944, 5801, 5673, 5313, 3264, 2986,
                  5499, 3084, 7636, 5650, 5517, 3220, 3117, 5617, 5928, 5554, 5687, 5503, 3015, 2939, 5960, 5690, 5750,
                  6016, 5453, 3265, 3062, 5786, 6213, 5902, 5906, 5292, 3097, 3029, 5688, 5817, 5718, 5755, 5373, 2999,
                  2959, 5664, 5743, 5712, 5755, 5426, 3304, 3120, 5666, 5738, 5493, 5816, 5246, 3228, 3283, 5638, 5857,
                  5782, 5922, 5649, 3288, 3343, 6299, 6107, 6193, 6435, 5837, 3263, 3378, 6344, 6024, 6240, 6013, 5627,
                  3333, 3367, 6133, 6114, 6012, 6287, 6180, 3409, 3432, 6161, 6206, 6350, 6112, 6197, 3555, 3442, 6275,
                  6370, 6770, 6689, 6552, 3376, 3489, 6772, 6748, 6659, 6754, 6493, 3982, 3719, 6749, 6517, 6597, 6524,
                  6395, 3747, 3187, 6858, 6354, 6434, 3409, 8686, 3330, 3294, 5740, 3959, 6280, 6623, 6327, 3526, 3469,
                  6440, 6381, 6443, 6408, 6119, 3565, 3392, 6555, 6257, 6706, 6222, 6264, 3407, 3183, 3826, 6424, 6658,
                  6568, 6065, 3427, 3466, 6558, 6653, 6452, 6501, 6111, 3463, 3570, 7366, 6591, 6780, 6640, 6094, 3542,
                  3405, 6405, 6352, 6586, 6537, 6166, 3290, 3335, 6531, 6227, 6547, 6873, 6060, 3276, 3110, 5582, 4824,
                  2704, 2692, 2878, 2922, 2947, 5052, 4750, 3887, 3123, 4821, 3291, 3293, 6550, 6636, 6599, 6601, 6410,
                  3862, 3759, 6684, 6589, 6446, 6498, 6193, 3416, 3459, 6626, 6615, 6625, 6869, 6544, 3709, 3701, 6870,
                  6586, 6838, 6765, 6657, 3882]

# 01/12/2014 - 31/01/2015
# Total: 328,468
DOCS_DEC_JAN = [7366, 6591, 6780, 6640, 6094, 3542, 3405, 6405, 6352, 6586, 6537, 6166, 3290, 3335, 6531, 6227, 6547,
                6873, 6060, 3276, 3110, 5582, 4824, 2704, 2692, 2878, 2922, 2947, 5052, 4750, 3887, 3123, 4821, 3291,
                3293, 6550, 6636, 6599, 6601, 6410, 3862, 3759, 6684, 6589, 6446, 6498, 6193, 3416, 3459, 6626, 6615,
                6625, 6869, 6544, 3709, 3701, 6870, 6586, 6838, 6765, 6657, 3882]

# Total: 2,078,774
FULL_TEXTS = np.array(
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
FULL_TEXTS_DEC_JAN = np.array(
    [7325, 6553, 6743, 6603, 6060, 3521, 3385, 6369, 6315, 6549, 6501, 6129, 3272, 3316, 6492, 6190, 6509, 6835, 6028,
     3256, 3092, 5550, 4796, 2688, 2677, 2862, 2904, 2928, 5022, 4722, 3863, 3106, 4794, 3272, 3276, 6513, 6598, 6561,
     6563, 6372, 3841, 3735, 6645, 6549, 6411, 6462, 6158, 3396, 3438, 6586, 6576, 6587, 6830, 6505, 3686, 3677, 6829,
     6547, 6797, 6726, 6616, 3858])


class FullTextDocument:
    __slots__ = ['name', 'date', 'category', 'abstract', 'text']

    def __init__(self):
        self.name = None
        self.date = None
        self.category = None
        self.abstract = None
        self.text = None


class CzechFullTexts:
    """
    Wrapper around the Czech full texts corpus. When iterated over, yields documents represented by the `Document`
    class. Can be queried for the relative days of the documents, in the same order as documents are yielded.
    :param names: whether to retrieve document headlines
    :param dates: whether to retrieve true publication dates, not just relative to start
    :param categories: whether to retrieve category marks
    :param abstracts: whether to retrieve document abstracts
    :param texts: whether to retrieve document texts
    """

    def __init__(self, dataset, names=None, dates=None, categories=None, abstracts=None, texts=True):
        if dataset == 'full':
            self.document_paths = os.listdir(CZECH_FULL_TEXTS)
            self.relative_days = FULL_TEXTS
        elif dataset == 'dec-jan':
            self.document_paths = CZECH_FULL_TEXTS_DEC_JAN
            self.relative_days = FULL_TEXTS_DEC_JAN
        else:
            raise ValueError('Unknown dataset - select either `full` or `dec-jan`.')

        self.names = names
        self.dates = dates
        self.categories = categories
        self.abstracts = abstracts
        self.texts = texts

    def __iter__(self):
        in_text = False

        for file in self.document_paths:
            with open(os.path.join(CZECH_FULL_TEXTS, file), 'r', encoding='utf8', errors='ignore') as f:
                loaded_text = []

                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        line = line.split(':')
                        key = line[0].strip()
                        val = ''.join(line[1:]).strip()

                        if key == '#name':
                            document = FullTextDocument()  # Start a new document when '#name' is encountered.
                            if self.names:
                                document.name = val
                        elif key == '#date' and self.dates:
                            document.date = val
                        elif key == '#category' and self.categories:
                            document.category = val
                        elif key == '#abstract' and self.abstracts:
                            document.abstract = val
                        elif key == '#text':
                            in_text = True
                        elif key == '#text.end':
                            if self.texts:
                                document.text = ''.join(loaded_text)
                                del loaded_text[:]
                            in_text = False
                        elif key == '#end#':  # Assume '#end#' ends the document.
                            yield document
                    elif not line:
                        continue  # Skip empty lines.
                    else:
                        if self.texts and in_text:
                            loaded_text.append(line)

    def fetch_relative_days(self):
        """
        Return a numpy array of relative days for the dataset. Implemented separately so the original texts can be
        comfortably streamed while keeping relative days in an array.
        :return: numpy array of relative days in the same order as the documents
        """
        return np.repeat(np.arange(self.relative_days.size), self.relative_days)


class TaggedDocument:
    __slots__ = ['name', 'text']

    def __init__(self):
        self.name = []
        self.text = []


class CzechTaggedTexts:
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

    def __init__(self, dataset, fetch_forms, pos=('A', 'C', 'D', 'I', 'J', 'N', 'P', 'V', 'R', 'T')):
        """
        Initialize the document fetcher which yields `TaggedDocument` objects upon iteration.
        :param dataset: which dataset to use (full/dec-jan)
        :param fetch_forms: whether to fetch forms od lemmas of the individual words
        :param pos: which POS categories to fetch
        """
        if dataset == 'full':
            self.document_paths = os.listdir(CZECH_TAGGED_TEXTS)
            self.relative_days = FULL_TEXTS
            raise NotImplementedError('Full tagged dataset not yet implemented')
        elif dataset == 'dec-jan':
            self.document_paths = CZECH_TAGGED_TEXTS_DEC_JAN
            self.relative_days = FULL_TEXTS_DEC_JAN
        else:
            raise ValueError('Unknown dataset - select either `full` or `dec-jan`.')

        self.fetch_forms = fetch_forms
        self.pos = frozenset(pos)

    def __iter__(self):
        in_title = False
        in_text = False

        fetch_forms = self.fetch_forms
        pos = self.pos

        for file in self.document_paths:
            with open(os.path.join(CZECH_TAGGED_TEXTS, file), 'r', encoding='utf8', errors='ignore') as f:
                for line in f:
                    line = line.strip()

                    if line.startswith('#'):
                        # Control lines
                        if line == '#docstart':
                            document = TaggedDocument()
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
                            logging.warning('Content line does not have 3 parts separated by tabs: %s', line)
                            continue

                        # Skip undesired POS tags.
                        if split[2][0] not in pos:
                            continue

                        if in_title:
                            if fetch_forms:
                                document.name.append(split[0])
                            else:
                                document.name.append(split[1])
                        elif in_text:
                            if fetch_forms:
                                document.text.append(split[0])
                            else:
                                document.text.append(split[1])

    def fetch_relative_days(self):
        """
        Return a numpy array of relative days for the dataset. Implemented separately so the original texts can be
        comfortably streamed while keeping relative days in an array.
        :return: numpy array of relative days in the same order as the documents
        """
        return np.repeat(np.arange(self.relative_days.size), self.relative_days)


def fetch_czech_corpus(num_docs):
    """
    Load documents from preprocessed text files.
    :param num_docs: the number of documents to load
    :return: a list of documents and a list of relative days
    """
    docs = []
    dates = []

    with open(CZECH_DATASET) as f:
        lines_read = 0

        for line in f:
            if lines_read == num_docs:
                break

            docs.append(line)
            lines_read += 1

    for i, num in enumerate(DOCS_IN_MONTHS):
        for j in range(num):
            dates.append(i)

    dates = dates[:num_docs]

    assert len(docs) == len(dates)

    return docs, np.array(dates)


def fetch_czech_corpus_dec_jan():
    """
    Load documents from preprocessed text files for the DEC-JAN period.
    :return: a list of documents and a list of relative days
    """
    dates = []

    with open(CZECH_DEC_JAN_DATASET) as f:
        docs = f.readlines()

    for i, num in enumerate(DOCS_DEC_JAN):
        for j in range(num):
            dates.append(i)

    assert len(docs) == len(dates)

    return docs, np.array(dates)
