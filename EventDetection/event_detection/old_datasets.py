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
