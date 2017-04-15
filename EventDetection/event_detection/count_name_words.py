import collections
import itertools
import logging
import pickle
from pprint import pprint

from event_detection import data_fetchers


def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    keyword_fetcher = data_fetchers.CzechLemmatizedTexts(dataset='full', fetch_forms=False, pos='NV', names_only=True)
    stream = itertools.chain.from_iterable(map(lambda doc: doc.name, keyword_fetcher))
    counter = collections.Counter(stream)

    with open('./name_word_counts.pickle', 'wb') as f:
        pickle.dump(counter, f)


def check():
    with open('./name_word_counts.pickle', 'rb') as f:
        counter = pickle.load(f)

    pprint(counter.most_common(1000))
    print(len(counter))


if __name__ == '__main__':
    # main()
    check()
