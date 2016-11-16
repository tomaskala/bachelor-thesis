import json
import os
import re
from datetime import datetime
from html.parser import HTMLParser

import numpy as np
from bs4 import BeautifulSoup

DE_NEWS_DATASET = '../../de-news'
SIGNAL_DATASET = '../../sample-1M.jsonl'
REUTERS_DATASET = '../../reuters'
CRAWL_DATASET = '../../crawl'
CZECH_DEC_JAN_DATASET = '../../out2DecJan.txt'
CZECH_DATASET = '../../out2.txt'


def fetch_signal_news(skip=0, num_docs=50000, news_only=True):
    """
    Fetch the given number of documents from the file and returns 3 lists. The first list represents the documents,
    the second represents their days relative to the earliest date, and the third contains True for those days
    falling on weekdays. Optionally skip the first n lines.
    The dataset contains 265,512 blog articles and 734,488 news articles. Blog posts can (and should!) be skipped.
    Source: http://research.signalmedia.co/newsir16/signal-dataset.html
    :param skip: skip this many lines from the beginning of the file
    :param num_docs: number of documents to fetch
    :param news_only: if True, skip blog articles
    :return: (documents, relative_days, weekdays) with relative_days[i] being the day of documents[i]
    """

    class HTMLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.reset()
            self.strict = False
            self.convert_charrefs = True
            self.fed = []

        def handle_data(self, d):
            self.fed.append(d)

        def get_data(self):
            return ''.join(self.fed)

        def reset(self):
            super().reset()
            self.fed = []

        def error(self, message):
            raise ValueError(message)

    docs = []
    dates = []
    earliest_date = datetime(year=9999, month=1, day=1)
    s = HTMLStripper()

    with open(SIGNAL_DATASET) as f:
        for i in range(skip):
            next(f)

        lines_read = 0

        for line in f:
            if lines_read == num_docs:
                break

            obj = json.loads(line)

            category = obj['media-type'].lower()

            if news_only and category == 'blog':
                continue

            date = datetime.strptime(obj['published'], '%Y-%m-%dT%H:%M:%SZ')

            # Strip the possible HTML tags. More efficient than Beautiful Soup.
            s.reset()
            s.feed(obj['content'])
            content = s.get_data()

            docs.append(content)
            dates.append(date)
            earliest_date = min(earliest_date, date)
            lines_read += 1

    relative_days, weekdays = transform_dates(dates, earliest_date)
    return docs, relative_days, weekdays


def fetch_de_news():
    """
    Fetch all documents from the DE News data set along with their publication dates and all weekdays out of those.
    Uses the same format as other document fetchers.
    Source: http://homepages.inf.ed.ac.uk/pkoehn/publications/de-news/
    :return: (documents, relative_days, weekdays) with relative_days[i] being the day of documents[i]
    """
    docs = []
    dates = []
    earliest_date = datetime(year=9999, month=1, day=1)
    datetime_regexp = re.compile('-(\d{4}-\d{2}-\d{2})-')

    for file in os.listdir(DE_NEWS_DATASET):
        with open(os.path.join(DE_NEWS_DATASET, file)) as f:
            doc = f.read()
            date = datetime.strptime(datetime_regexp.findall(file)[0], '%Y-%m-%d')

            docs.append(doc)
            dates.append(date)
            earliest_date = min(earliest_date, date)

    relative_days, weekdays = transform_dates(dates, earliest_date)
    return docs, relative_days, weekdays


def fetch_reuters():
    """
    Fetch all documents from the Reuters 21578 dataset which have meaningful content along with their publication
    dates and all weekdays out of those. Uses the same format as other fetchers.
    Source: https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/
    :return: (documents, relative_days, weekdays) with relative_days[i] being the day of documents[i]
    """
    docs = []
    dates = []
    earliest_date = datetime(year=9999, month=1, day=1)
    datetime_regexp = re.compile(r'\s*(\d{1,2}-\w{3}-\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{1,5})', re.I)

    for file in filter(lambda filename: filename.endswith('.sgm'), os.listdir(REUTERS_DATASET)):
        with open(os.path.join(REUTERS_DATASET, file)) as f:
            content = f.read().replace('<BODY>', '<CONTENT>').replace('</BODY>', '</CONTENT>')
            soup = BeautifulSoup(content, 'lxml')

            reuters = soup.findAll('reuters')

            for reuter in reuters:
                if 'content' not in map(lambda c: c.name, reuter.findChildren()):
                    continue

                date = reuter.find('date')

                date_str = date.text
                match = datetime_regexp.match(date_str).group(1)

                fulldate = datetime.strptime(match, '%d-%b-%Y %H:%M:%S.%f')
                dates.append(fulldate)
                earliest_date = min(earliest_date, fulldate)

                content = reuter.find('content')
                docs.append(content.text)

    relative_days, weekdays = transform_dates(dates, earliest_date)
    return docs, relative_days, weekdays


def fetch_crawl_data(skip=0, num_docs=50000):
    """
    Source: https://www.cs.washington.edu/node/9473/
    :param skip: number of documents to skip
    :param num_docs: number of documents to fetch
    :return:
    """
    docs = []
    dates = []
    earliest_date = datetime(year=9999, month=1, day=1)

    with open(CRAWL_DATASET, encoding='utf8') as f:
        for i in range(skip):
            next(f)

        lines_read = 0

        for line in f:
            if lines_read == num_docs:
                break

            obj = json.loads(line)

            if 'text' not in obj.keys() or 'date' not in obj.keys():
                continue

            date = datetime.strptime(obj['date'], '%b %d, %Y %I:%M:%S %p')
            content = obj['text']

            docs.append(content)
            dates.append(date)
            earliest_date = min(earliest_date, date)
            lines_read += 1

    relative_days, weekdays = transform_dates(dates, earliest_date)
    return docs, relative_days, weekdays


def fetch_czech_corpus_dec_jan():
    dates = []
    # 01/12/2014 - 31/01/2015
    doc_num_dec_jan = [7366, 6591, 6780, 6640, 6094, 3542, 3405, 6405, 6352, 6586, 6537, 6166, 3290, 3335, 6531, 6227,
                       6547, 6873, 6060, 3276, 3110, 5582, 4824, 2704, 2692, 2878, 2922, 2947, 5052, 4750, 3887, 3123,
                       4821, 3291, 3293, 6550, 6636, 6599, 6601, 6410, 3862, 3759, 6684, 6589, 6446, 6498, 6193, 3416,
                       3459, 6626, 6615, 6625, 6869, 6544, 3709, 3701, 6870, 6586, 6838, 6765, 6657, 3882]

    with open(CZECH_DEC_JAN_DATASET) as f:
        docs = f.readlines()

    for i, num in enumerate(doc_num_dec_jan):
        for j in range(num):
            dates.append(i)

    assert len(docs) == len(dates)

    return docs, dates


def fetch_czech_corpus(num_docs=50000):
    docs = []
    dates = []
    doc_num = [3002, 5442, 5559, 3059, 3163, 6089, 6013, 6284, 5918, 5916, 3221, 3304, 6139, 5978, 6293, 6103, 5821,
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

    with open(CZECH_DATASET) as f:
        lines_read = 0

        for line in f:
            if lines_read == num_docs:
                break

            docs.append(line)
            lines_read += 1

    for i, num in enumerate(doc_num):
        for j in range(num):
            dates.append(i)

    dates = dates[:num_docs]

    assert len(docs) == len(dates)

    return docs, dates


def transform_dates(dates, earliest_date):
    """
    Collect relative days to the earliest date into one list, and days falling on weekdays into the second.
    This information is later utilized by heuristic stopwords detection.
    :param dates: list of dates retrieved from the file
    :param earliest_date: minimum from the dates argument
    :return: list of days relative to the earliest date and a boolean list of whether each day from [0, latest] falls
        on weekday or not
    """
    relative_days = []
    weekdays = set()

    for date in dates:
        relative_day = (date - earliest_date).days
        relative_days.append(relative_day)

        if 0 <= date.weekday() <= 4:
            weekdays.add(relative_day)

    return relative_days, np.array([True if day in weekdays else False for day in range(max(relative_days) + 1)])
