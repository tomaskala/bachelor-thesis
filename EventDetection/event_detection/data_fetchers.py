import os

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
# Total: 2090635
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
# Total: 328468
DOCS_DEC_JAN = [7366, 6591, 6780, 6640, 6094, 3542, 3405, 6405, 6352, 6586, 6537, 6166, 3290, 3335, 6531, 6227,
                6547, 6873, 6060, 3276, 3110, 5582, 4824, 2704, 2692, 2878, 2922, 2947, 5052, 4750, 3887, 3123,
                4821, 3291, 3293, 6550, 6636, 6599, 6601, 6410, 3862, 3759, 6684, 6589, 6446, 6498, 6193, 3416,
                3459, 6626, 6615, 6625, 6869, 6544, 3709, 3701, 6870, 6586, 6838, 6765, 6657, 3882]


class Document:
    __slots__ = ['name', 'date', 'category', 'abstract', 'text']

    def __init__(self):
        self.name = None
        self.date = None
        self.category = None
        self.abstract = None
        self.text = None


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

    return docs, dates


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

    return docs, dates


def fetch_czech_full_texts(names=None, dates=None, categories=None, abstracts=None, texts=True):
    """
    Load documents from full text files. The documents are represented by the `Document` object containing only
    those values chosen to retrieve, and None everywhere else. The documents are yielded to make the process
    as memory efficient as possible. Setting an argument to `True` will mark that field for retrieval.

    :param names: whether to retrieve document headlines
    :param dates: whether to retrieve true publication dates, not just relative to start
    :param categories: whether to retrieve category marks
    :param abstracts: whether to retrieve document abstracts
    :param texts: whether to retrieve document texts
    :return: yields (relative_day, document) pairs -- split it by `dates, docs = zip(*load_full_texts))`
    """
    yield from _load_full_texts(os.listdir(CZECH_FULL_TEXTS), names, dates, categories, abstracts, texts)


def fetch_czech_full_texts_dec_jan(names=None, dates=None, categories=None, abstracts=None, texts=True):
    """
    Load documents from full text files for the DEC-JAN period. The documents are represented by the `Document`
    object containing only those values chosen to retrieve, and None everywhere else. The documents are yielded
    to make the process as memory efficient as possible. Setting an argument to `True` will mark that field for
    retrieval.

    :param names: whether to retrieve document headlines
    :param dates: whether to retrieve true publication dates, not just relative to start
    :param categories: whether to retrieve category marks
    :param abstracts: whether to retrieve document abstracts
    :param texts: whether to retrieve document texts
    :return: yields (relative_day, document) pairs -- split it by `dates, docs = zip(*load_full_texts))`
    """
    yield from _load_full_texts(CZECH_FULL_TEXTS_DEC_JAN, names, dates, categories, abstracts, texts)


def _load_full_texts(files, names, dates, categories, abstracts, texts):
    in_text = False

    for relative_day, file in enumerate(files):
        with open(os.path.join(CZECH_FULL_TEXTS, file), 'r', encoding='utf8', errors='ignore') as f:
            loaded_text = []

            for line in f:
                line = line.strip()

                if line.startswith('#'):
                    line = line.split(':')
                    key = line[0].strip()
                    val = ''.join(line[1:]).strip()

                    if key == '#name':
                        document = Document()  # Start a new document when '#name' is encountered.
                        if names:
                            document.name = val
                    elif key == '#date' and dates:
                        document.date = val
                    elif key == '#category' and categories:
                        document.category = val
                    elif key == '#abstract' and abstracts:
                        document.abstract = val
                    elif key == '#text':
                        in_text = True
                    elif key == '#text.end':
                        if texts:
                            document.text = ''.join(loaded_text)
                            del loaded_text[:]
                        in_text = False
                    elif key == '#end#':
                        yield document, relative_day
                elif not line:
                    continue  # Skip empty lines.
                else:
                    if texts and in_text:
                        loaded_text.append(line)
