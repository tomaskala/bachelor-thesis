import argparse
import logging
import os
import sys


def check_gensim():
    dir_path = './gensim'
    necessary_files = [
        'word2vec_lemma_skipgram',
        'word2vec_lemma_skipgram.syn1neg.npy',
        'word2vec_lemma_skipgram.wv.syn0.npy']

    return os.path.exists(dir_path) and all(
        os.path.exists(os.path.join(dir_path, filename)) for filename in necessary_files)


def check_pickle():
    dir_path = './pickle'
    necessary_files = [
        'events_clusters.pickle',
        'event_docids_clusters.pickle',
        'event_docids_greedy.pickle',
        'event_docids_original.pickle',
        'event_summ_docs_clusters.pickle',
        'event_summ_docs_greedy.pickle',
        'event_summ_docs_original.pickle',
        'id2word.pickle',
        'id2word_original.pickle',
        'relative_days.pickle',
        'relative_days_original.pickle',
        'term_document.npz',
        'term_document_original.npz']

    return os.path.exists(dir_path) and all(
        os.path.exists(os.path.join(dir_path, filename)) for filename in necessary_files)


def main():
    sys.path.insert(0, '../')
    sys.path.insert(0, './')

    if not check_gensim() or not check_pickle():
        print('Please unzip the data from the DVD to the current directory.', file=sys.stderr)
        exit(1)

    parser = argparse.ArgumentParser(
        description='Run the event detection from the serialized data. This will produce a directory with event graphs'
                    ' and a file with event summaries. The process can take quite a bit of time, see the Evaluation'
                    ' chapter of the thesis for approximate times.')
    parser.add_argument('-m', '--method', choices=['original', 'embedded-greedy', 'cluster-based'],
                        help='Method to use for event detection.', required=True)
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    if args.method == 'original':
        from event_detection import original_method
        original_method.main()
    elif args.method == 'embedded-greedy':
        from event_detection import event_detector
        event_detector.main(cluster_based=False)
    else:
        from event_detection import event_detector
        event_detector.main(cluster_based=True)


if __name__ == '__main__':
    main()
