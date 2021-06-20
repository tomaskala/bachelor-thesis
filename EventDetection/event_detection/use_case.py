import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from event_detection.k_medoids import KMedoids
from event_detection.event_detector import calculate_trajectory_cutoff, WINDOW
from event_detection.postprocessing import estimate_distribution_periodic

PICKLE_PATH = '../event_detection/pickle'
EVENTS_PATH = os.path.join(PICKLE_PATH, 'events_clusters.pickle')
ID2WORD_PATH = os.path.join(PICKLE_PATH, 'id2word.pickle')
EVENTS_WMDISTANCE = os.path.join(PICKLE_PATH, 'events_wmdistance.pickle')
CRIME_TRAJECTORIES = os.path.join(PICKLE_PATH, 'events_crime.pickle')
WEATHER_TRAJECTORIES = os.path.join(PICKLE_PATH, 'events_weather.pickle')
BOW_MATRIX_PATH = os.path.join(PICKLE_PATH, 'term_document.npz')
RELATIVE_DAYS_PATH = os.path.join(PICKLE_PATH, 'relative_days.pickle')

GENSIM_OUT = './gensim'
WORD2VEC_PATH = os.path.join(GENSIM_OUT, './word2vec_lemma_skipgram')


def main():
    with open(EVENTS_PATH, mode='rb') as f:
        events = pickle.load(f)

    with open(ID2WORD_PATH, mode='rb') as f:
        id2word = pickle.load(f)

    event_words = [[id2word[word_id] for word_id in event] for event in events]

    # word2vec_model = gensim.models.Word2Vec.load(WORD2VEC_PATH)
    # distance_matrix = np.zeros(shape=(len(events), len(events)), dtype=float)
    #
    # for i in range(len(events)):
    #     for j in range(i + 1, len(events)):
    #         distance_matrix[i, j] = word2vec_model.wmdistance(event_words[i], event_words[j])
    #         distance_matrix[j, i] = distance_matrix[i, j]
    #
    # with open(EVENTS_WMDISTANCE, mode='wb') as f:
    #     pickle.dump(distance_matrix, f)

    with open(EVENTS_WMDISTANCE, mode='rb') as f:
        distance_matrix = pickle.load(f)

    # With n_clusters=12, cluster #2 could be considered crime, cluster #4 weather.
    n_clusters = 12
    clusterer = KMedoids(n_clusters=n_clusters, distance_metric='precomputed', max_iter=1000)
    adjacency = clusterer.fit_predict(distance_matrix)

    clusters = [[] for _ in range(n_clusters)]

    for i, cluster_id in enumerate(adjacency):
        clusters[cluster_id].append(i)

    for i, cluster in enumerate(clusters):
        print(i)
        for event_id in cluster:
            print('\t', event_id, event_words[event_id])
        print()

    crime_cluster = clusters[2]
    weather_cluster = clusters[4]

    with open(CRIME_TRAJECTORIES, mode='rb') as f:
        crime_trajectories = pickle.load(f)

    cutoff = calculate_trajectory_cutoff(crime_trajectories, WINDOW)
    crime_trajectories[crime_trajectories <= cutoff] = 0.0

    n_days = crime_trajectories.shape[1]
    days = np.arange(n_days)

    avg_crime = np.mean(crime_trajectories, axis=0)

    with open(WEATHER_TRAJECTORIES, mode='rb') as f:
        weather_trajectories = pickle.load(f)

    cutoff = calculate_trajectory_cutoff(weather_trajectories, WINDOW)
    weather_trajectories[weather_trajectories <= cutoff] = 0.0

    avg_weather = np.mean(weather_trajectories, axis=0)

    # use_this_cluster = crime_cluster
    # use_these_trajectories = crime_trajectories
    # use_this_avg = avg_crime
    # use_this_period = 66

    use_this_cluster = weather_cluster
    use_these_trajectories = weather_trajectories
    use_this_avg = avg_weather
    use_this_period = 132

    event_parameters = estimate_distribution_periodic(use_this_avg, use_this_period)

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Event trajectories')
    plt.xlim(0.0, n_days)
    plt.grid(True)

    for event_id, trajectory in zip(use_this_cluster, use_these_trajectories):
        plt.plot(days, trajectory, label=', '.join([id2word[word_id] for word_id in events[event_id]]))

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('Resulting trajectory')
    plt.xlim(0.0, n_days)
    plt.grid(True)

    plt.plot(days, use_this_avg)

    xticks_pos = []
    xticks = []

    for mean, std in event_parameters:
        burst_start = int(max(np.floor(mean - std), 0))
        burst_end = int(min(np.ceil(mean + std), n_days - 1))
        burst_loc = (burst_start + burst_end) / 2

        xticks_pos.extend([burst_loc])
        xticks.extend([('{:.01f}'.format(burst_loc))])

        where = np.zeros(n_days, dtype=bool)
        where[burst_start:burst_end + 1] = True
        plt.fill_between(days, use_this_avg, where=where, color='#2b93db')

    plt.xticks(xticks_pos, xticks)

    plt.xlabel('Days (relative to 1/1/2014)')
    plt.ylabel('DFIDF')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    main()
