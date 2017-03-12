import logging

import data_fetchers, preprocessing


DATASET = 'full'
POS_EMBEDDINGS = ('A', 'C', 'D', 'I', 'J', 'N', 'P', 'V', 'R', 'T')


def main():
    # Set up logging
    logger = logging.getLogger()
    handler = logging.FileHandler('./word2vec_log.log')
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Create fetcher
    embedding_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=DATASET, fetch_forms=False, pos=POS_EMBEDDINGS)
    w2v_model = preprocessing.perform_word2vec(embedding_fetcher)
    print('Done')


if __name__ == '__main__':
    main()
