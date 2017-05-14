import logging
import os
import pickle
from time import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture as gmm
import wordcloud
from scipy.optimize import curve_fit
from scipy.stats import cauchy, norm
from sklearn.neighbors import NearestNeighbors

import event_detection.postprocessing as post
from event_detection import annotations, data_fetchers

matplotlib.rc('font', family='DejaVu Sans')


def moving_average(vector, window):
    """
    Compute the moving average along the given vector using a window of the given length.
    :param vector: the vector whose moving average to compute
    :param window: length of the window to use in the computation
    :return: moving average of length len(vector) - window + 1
    """
    weights = np.ones(window) / window
    return np.convolve(vector, weights, 'valid')


def visualise_clusters(clusters, documents, output_dir='./wordcloud'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cluster_words = [[] for _ in range(len(clusters))]

    for i, cluster in enumerate(clusters):
        for doc in cluster:
            cluster_words[i].extend(documents[doc].split(r'\s'))

    for i, words in enumerate(cluster_words):
        wc = wordcloud.WordCloud(width=2000, height=1200).generate(' '.join(words))
        fig = plt.figure()
        fig.set_size_inches(20, 12)
        plt.imshow(wc)
        plt.axis('off')
        wc.to_file(os.path.join(output_dir, '{0:02d}.png'.format(i)))


def output_events(events, events_docids_repr, id2word, doc2vec_model, num_aperiodic, aperiodic_path, periodic_path,
                  cluster_based):
    from event_detection import event_detector

    full_fetcher = data_fetchers.CzechLemmatizedTexts(dataset=event_detector.DATASET, fetch_forms=True,
                                                      pos=event_detector.POS_EMBEDDINGS + ('Z', 'X'))

    # [[(burst_start, burst_end, [annotations.LemmatizedDocument] ... event_bursts] ... events]
    EVENT_FULL_DOCS_GREEDY_PATH = os.path.join(event_detector.PICKLE_PATH, 'event_full_docs_greedy.pickle')
    EVENT_FULL_DOCS_CLUSTERS_PATH = os.path.join(event_detector.PICKLE_PATH, 'event_full_docs_clusters.pickle')

    if cluster_based:
        if os.path.exists(EVENT_FULL_DOCS_CLUSTERS_PATH):
            logging.info('Deserializing full documents.')

            with open(EVENT_FULL_DOCS_CLUSTERS_PATH, mode='rb') as f:
                events_docs_repr = pickle.load(f)

            logging.info('Deserialized full documents.')
        else:
            logging.info('Retrieving full documents.')
            t = time()

            events_docs_repr = annotations.docids2documents(events_docids_repr, full_fetcher)

            with open(EVENT_FULL_DOCS_CLUSTERS_PATH, mode='wb') as f:
                pickle.dump(events_docs_repr, f)

            logging.info('Retrieved and serialized full documents in %fs.', time() - t)
    else:
        if os.path.exists(EVENT_FULL_DOCS_GREEDY_PATH):
            logging.info('Deserializing full documents.')

            with open(EVENT_FULL_DOCS_GREEDY_PATH, mode='rb') as f:
                events_docs_repr = pickle.load(f)

            logging.info('Deserialized full documents.')
        else:
            logging.info('Retrieving full documents.')
            t = time()

            events_docs_repr = annotations.docids2documents(events_docids_repr, full_fetcher)

            with open(EVENT_FULL_DOCS_GREEDY_PATH, mode='wb') as f:
                pickle.dump(events_docs_repr, f)

            logging.info('Retrieved and serialized full documents in %fs.', time() - t)

    aperiodic_events = events[:num_aperiodic]
    periodic_events = events[num_aperiodic:]

    aperiodic_events_docs_repr = events_docs_repr[:num_aperiodic]
    periodic_events_docs_repr = events_docs_repr[num_aperiodic:]

    output_events_inner(aperiodic_events_docs_repr, aperiodic_events, id2word, doc2vec_model, dirname=aperiodic_path)
    output_events_inner(periodic_events_docs_repr, periodic_events, id2word, doc2vec_model, dirname=periodic_path)


def output_events_inner(events_docs_repr, events, id2word, doc2vec_model, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'events.txt'), 'w') as f:
        for i, event in enumerate(events_docs_repr):
            event_keywords = [id2word[keyword_id] for keyword_id in events[i]]
            print('Event {:03d}: [{:s}]'.format(i, ', '.join(event_keywords)), file=f)

            for burst in event:
                burst_start, burst_end, burst_docs = burst
                doc_vecs = doc2vec_model.docvecs[[doc.doc_id.item() for doc in burst_docs]]
                mean_vec = np.mean(doc_vecs, axis=0)
                neighbors = NearestNeighbors(n_neighbors=1)
                neighbors.fit(doc_vecs)
                centroid_id = neighbors.kneighbors(np.array([mean_vec]), return_distance=False)
                centroid_doc = burst_docs[centroid_id]

                print('Burst ({:d} - {:d}) with {:d} documents'.format(burst_start, burst_end, len(burst_docs)), file=f)
                print('Most similar headline: "{:s}" (document #{:d})'.format(' '.join(burst_docs[0].name),
                                                                              burst_docs[0].doc_id), file=f)
                print('Centroid headline: "{:s}" (document #{:d})'.format(' '.join(centroid_doc.name),
                                                                          centroid_doc.doc_id),
                      file=f)
                print('Least similar headline: "{:s}" (document #{:d})'.format(' '.join(burst_docs[-1].name),
                                                                               burst_docs[-1].doc_id), file=f)
                print(file=f)

            if len(event) > 1:
                print(file=f)


def plot_events(feature_trajectories, events, id2word, dps, dirname='../events'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    n_days = feature_trajectories.shape[1]
    days = np.arange(feature_trajectories.shape[1])
    event_trajectories, event_periods = post.create_events_trajectories(events, feature_trajectories, dps)

    for i, (event, event_trajectory, event_period) in enumerate(zip(events, event_trajectories, event_periods)):
        fig = plt.figure()

        plt.subplot(2, 1, 1)
        plt.title('Event features')
        plt.xlim(0.0, n_days)
        plt.grid(True)

        for feature in event:
            y = feature_trajectories[feature]
            plt.plot(days, y, label=id2word[feature])

            plt.xlabel('Days')
            plt.ylabel('DFIDF')
            plt.legend()

        plt.subplot(2, 1, 2)
        plt.title('Event trajectory')
        plt.xlim(0.0, n_days)
        plt.grid(True)
        plt.plot(days, event_trajectory, label='Period: {:d}'.format(event_period), color='red', linewidth=1.5)

        if event_period > np.ceil(n_days / 2):
            # Aperiodic
            mean, std = post.estimate_distribution_aperiodic(event_trajectory)

            burst_start = int(max(np.floor(mean - std), 0))
            burst_end = int(min(np.ceil(mean + std), n_days - 1))
            burst_loc = (burst_start + burst_end) / 2

            xticks_pos = [burst_start, burst_loc, burst_end]
            xticks = [('{:d}'.format(burst_start)), ('{:.01f}'.format(burst_loc)), ('{:d}'.format(burst_end))]
            plt.axvline(burst_start, color='b')
            plt.axvline(burst_end, color='b')
            plt.xticks(xticks_pos, xticks)
        else:
            # Periodic
            params = post.estimate_distribution_periodic(event_trajectory, event_period)

            xticks_pos = []
            xticks = []

            for mean, std in params:
                burst_start = int(max(np.floor(mean - std), 0))
                burst_end = int(min(np.ceil(mean + std), n_days - 1))
                burst_loc = (burst_start + burst_end) / 2

                xticks_pos.extend([burst_start, burst_loc, burst_end])
                xticks.extend([('{:d}'.format(burst_start)), ('{:.01f}'.format(burst_loc)), ('{:d}'.format(burst_end))])
                plt.axvline(burst_start, color='b')
                plt.axvline(burst_end, color='b')

            plt.xticks(xticks_pos, xticks)

        plt.xlabel('Days')
        plt.ylabel('DFIDF')
        plt.legend()

        fig.set_size_inches(16, 10)
        plt.tight_layout()
        fig.savefig(os.path.join(dirname, '{:03d}.png'.format(i)))
        plt.close(fig)
        plt.clf()


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def plot_events_to_eps(trajectories, events, id2word, dps):
    dirname = '../EVENTS_OUT_BIG'

    # font = {'family': 'normal',
    #         'size': 12}

    # matplotlib.rc('font', **font)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    events_indices = [11, 41, 39, 63, 42, 16, 18]  # For clusters
    events_indices = [17, 19, 54]  # For greedy
    e = [events[i] for i in events_indices]
    events = e
    n_days = trajectories.shape[1]
    days = np.arange(n_days)
    event_trajectories, event_periods = post.create_events_trajectories(events, trajectories, dps)

    from scipy.signal import periodogram
    freqs, pgram = periodogram(trajectories)

    dps_indices = np.argmax(pgram, axis=1)
    feature_indices = np.arange(trajectories.shape[0])

    dps = pgram[feature_indices, dps_indices]

    with np.errstate(divide='ignore'):
        ffreqs = np.tile(freqs, (trajectories.shape[0], 1))

    df = ffreqs[feature_indices, dps_indices]

    for i, event, event_trajectory, event_period in zip(events_indices, events, event_trajectories, event_periods):
        # 1. Event keywords
        fig = plt.figure()
        plt.xlim(0.0, n_days)

        for feature in event:
            y = trajectories[feature]
            plt.plot(days, y, label=id2word[feature])

        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_words.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # 2. Event trajectory
        fig = plt.figure()
        plt.xlim(0.0, n_days)
        plt.plot(days, event_trajectory)
        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_trajectory.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # 3. Event trajectory with bursty periods
        fig = plt.figure()
        plt.xlim(0.0, n_days)
        plt.plot(days, event_trajectory)

        if event_period > np.ceil(n_days / 2):
            # Aperiodic
            mean, std = post.estimate_distribution_aperiodic(event_trajectory)

            burst_start = int(max(np.floor(mean - std), 0))
            burst_end = int(min(np.ceil(mean + std), n_days - 1))
            burst_loc = (burst_start + burst_end) / 2

            xticks_pos = [burst_loc]
            xticks = [('{:.01f}'.format(burst_loc))]
            plt.axvline(burst_start, color='b')
            plt.axvline(burst_end, color='b')
            plt.xticks(xticks_pos, xticks)
        else:
            # Periodic
            params = post.estimate_distribution_periodic(event_trajectory, event_period)

            xticks_pos = []
            xticks = []

            for mean, std in params:
                burst_start = int(max(np.floor(mean - std), 0))
                burst_end = int(min(np.ceil(mean + std), n_days - 1))
                burst_loc = (burst_start + burst_end) / 2

                xticks_pos.extend([burst_loc])
                xticks.extend([('{:.01f}'.format(burst_loc))])

                where = np.zeros(n_days, dtype=bool)
                where[burst_start:burst_end + 1] = True
                plt.fill_between(days, event_trajectory, where=where, color='#2b93db')

            plt.xticks(xticks_pos, xticks)

        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_trajectory_bursts.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # 4. Event trajectory with cutoff
        fig = plt.figure()
        plt.xlim(0.0, n_days)
        plt.plot(days, event_trajectory)

        ma = moving_average(event_trajectory, 7)
        ma_mean = np.mean(ma)
        ma_std = np.std(ma)
        cutoff = ma_mean + ma_std

        plt.hlines(cutoff, 0, n_days, colors='red', linestyles='dashed', linewidth=1.5)

        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_trajectory_cutoff.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # 5. Periodogram
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.xlim(0.0, freqs[-1])
        ax.plot(freqs, pgram[i])

        plt.xlabel('Frequency')
        plt.ylabel('Periodogram')
        plt.legend()
        plt.tight_layout()
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_periodogram.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        # 6. Trajectory with density fit.
        fig = plt.figure()
        plt.xlim(0.0, n_days)
        x = np.linspace(0.0, n_days, 1000)
        normalized_trajectory = event_trajectory / np.sum(event_trajectory)

        if event_period > np.ceil(n_days / 2):
            # Aperiodic
            mean, std = post.estimate_distribution_aperiodic(event_trajectory)

            def gaussian_curve(value, loc, scale):
                return norm.pdf(value, loc=loc, scale=scale)

            burst_start = int(max(np.floor(mean - std), 0))
            burst_end = int(min(np.ceil(mean + std), n_days - 1))
            burst_loc = (burst_start + burst_end) / 2

            pdf = gaussian_curve(x, mean, std)
            plt.plot(days, normalized_trajectory)
            plt.plot(x, pdf, 'r')

            where = np.zeros(len(x), dtype=bool)
            where[(x >= burst_start) & (x <= burst_end)] = True
            plt.fill_between(x, pdf, where=where, color='#ff7f7f')

            xticks_pos = [burst_loc]
            xticks = ['{:.01f}'.format(mean)]
            plt.xticks(xticks_pos, xticks)
        else:
            # Periodic
            params = post.estimate_distribution_periodic(event_trajectory, event_period)

            xticks_pos = []
            xticks = []
            where = np.zeros(len(x), dtype=bool)

            for mean, std in params:
                burst_start = int(max(np.floor(mean - std), 0))
                burst_end = int(min(np.ceil(mean + std), n_days - 1))
                burst_loc = (burst_start + burst_end) / 2

                xticks_pos.append(burst_loc)
                xticks.append('{:.01f}'.format(mean))

                where[(x >= burst_start) & (x <= burst_end)] = True

            plt.xticks(xticks_pos, xticks)

            observations = np.hstack((days.reshape(-1, 1), event_trajectory.reshape(-1, 1)))
            observations = observations[observations[:, 1] > cutoff, :]

            n_components = int(min(np.floor(n_days / event_period), len(observations)))
            g = gmm.GaussianMixture(n_components=n_components, covariance_type='diag')
            g.fit(observations)

            components = np.squeeze(np.array(
                [cauchy.pdf(x, mean[0], np.sqrt(2 * np.log(2)) * np.sqrt(cov[0])) for mean, cov in
                 zip(g.means_, g.covariances_)]))

            pdf = g.weights_ @ components
            plt.fill_between(x, pdf, where=where, color='#ff7f7f')

            plt.plot(days, normalized_trajectory)
            plt.plot(x, pdf, 'r')

        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}_density_fit.eps'.format(i)), format='eps', dpi=1200,
                    bbox_inches='tight')
        plt.close(fig)
        plt.clf()


def plot_original_event(trajectories, word2id):
    dirname = '../EVENTS_OUT'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # font = {'family': 'normal',
    #         'size': 12}
    #
    # matplotlib.rc('font', **font)

    words = ['palestinský', 'izraelský', 'Palestinec', 'Izrael']

    fig = plt.figure()
    n_days = trajectories.shape[1]
    days = np.arange(n_days)

    for word in words:
        idx = word2id[word]

        plt.plot(days, trajectories[idx], label=word, linewidth=1)

    plt.xlim(0.0, n_days)
    plt.xlabel('Days (relative to 1/1/2014)')
    plt.ylabel('DFIDF')
    plt.legend()
    plt.tight_layout()

    fig.set_size_inches(cm2inch(14, 8))
    fig.savefig(os.path.join(dirname, 'original_event.eps'), format='eps', dpi=1200, bbox_inches='tight')


def plot_greedy_event(trajectories, word2id):
    dirname = '../EVENTS_OUT'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # font = {'family': 'normal',
    #         'size': 12}
    #
    # matplotlib.rc('font', **font)

    words = ['sestřelit', 'raketa', 'Izrael', 'izraelský']

    fig = plt.figure()
    n_days = trajectories.shape[1]
    days = np.arange(n_days)

    for word in words:
        idx = word2id[word]

        plt.plot(days, trajectories[idx], label=word, linewidth=1)

    plt.xlim(0.0, n_days)
    plt.xlabel('Days (relative to 1/1/2014)')
    plt.ylabel('DFIDF')
    plt.legend()
    plt.tight_layout()

    fig.set_size_inches(cm2inch(14, 8))
    fig.savefig(os.path.join(dirname, 'greedy_event.eps'), format='eps', dpi=1200, bbox_inches='tight')


def plot_cluster_event(trajectories, word2id):
    dirname = '../EVENTS_OUT'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # font = {'family': 'normal',
    #         'size': 12}
    #
    # matplotlib.rc('font', **font)

    words = ['Gaza', 'Hamas', 'Izrael', 'Izraelec', 'Palestinec', 'izraelský', 'palestinský']

    fig = plt.figure()
    n_days = trajectories.shape[1]
    days = np.arange(n_days)

    for word in words:
        idx = word2id[word]

        plt.plot(days, trajectories[idx], label=word, linewidth=1)

    plt.xlim(0.0, n_days)
    plt.xlabel('Days (relative to 1/1/2014)')
    plt.ylabel('DFIDF')
    plt.legend()
    plt.tight_layout()

    fig.set_size_inches(cm2inch(14, 8))
    fig.savefig(os.path.join(dirname, 'cluster_event.eps'), format='eps', dpi=1200, bbox_inches='tight')


def plot_aperiodic_words(trajectories, dps, dp, dps_boundary, stream_length, id2word, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    font = {'family': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)

    # aperiodic_word_indices = np.where((dps > dps_boundary) & (dp > np.floor(stream_length / 2)))[0]
    aperiodic_word_indices = [64746, 71593, 63608, 84358, 105282, 21255, 60141]
    print('Plotting {:d} aperiodic words'.format(len(aperiodic_word_indices)))

    n_days = trajectories.shape[1]
    days = np.arange(n_days)

    with open(os.path.join(dirname, 'id2word.txt'), 'w', encoding='utf8') as f:
        for i in aperiodic_word_indices:
            print('{:d}: {:s}'.format(i, id2word[i]), file=f)

    for i in aperiodic_word_indices:
        fig = plt.figure()
        word_trajectory = trajectories[i]

        if i == 60141:
            label = '{:s} ({:s})'.format(id2word[i], 'Christmas')
        else:
            label = id2word[i]

        plt.xlim(0.0, n_days)
        plt.plot(days, word_trajectory, label=label)
        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        # fig.savefig(os.path.join(dirname, '%s_%d.png' % (id2word[i], i)))
        plt.close(fig)
        plt.clf()

    print('Plotting finished')


def plot_periodic_words(trajectories, dps, dp, dps_boundary, stream_length, id2word, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    font = {'family': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)

    # periodic_word_indices = np.where((dps > dps_boundary) & (dp <= np.floor(stream_length / 2)))[0]
    periodic_word_indices = [152999, 84055, 116963, 91143, 144057, 139919]
    print('Plotting {:d} periodic words'.format(len(periodic_word_indices)))

    n_days = trajectories.shape[1]
    days = np.arange(n_days)

    with open(os.path.join(dirname, 'id2word.txt'), 'w', encoding='utf8') as f:
        for i in periodic_word_indices:
            print('{:d}: {:s}'.format(i, id2word[i]), file=f)

    for i in periodic_word_indices:
        fig = plt.figure()
        word_trajectory = trajectories[i]

        if i == 116963:
            label = '{:s} ({:s})'.format(id2word[i], 'Friday')
        elif i == 91143:
            label = '{:s} ({:s})'.format(id2word[i], 'airplane')
        else:
            label = id2word[i]

        plt.xlim(0.0, n_days)
        plt.plot(days, word_trajectory, label=label)
        plt.xlabel('Days (relative to 1/1/2014)')
        plt.ylabel('DFIDF')
        plt.legend()
        plt.tight_layout()

        fig.set_size_inches(cm2inch(14, 8))
        fig.savefig(os.path.join(dirname, '{:d}.eps'.format(i)), format='eps', dpi=1200, bbox_inches='tight')
        # fig.savefig(os.path.join(dirname, '%s_%d_%d.png' % (id2word[i], i, dp[i])))
        plt.close(fig)
        plt.clf()

    from scipy.signal import periodogram
    freqs, pgram = periodogram(trajectories)

    dps_indices = np.argmax(pgram, axis=1)
    feature_indices = np.arange(trajectories.shape[0])

    dps = pgram[feature_indices, dps_indices]

    with np.errstate(divide='ignore'):
        ffreqs = np.tile(freqs, (trajectories.shape[0], 1))

    df = ffreqs[feature_indices, dps_indices]

    fig = plt.figure()

    plt.xlim(0.0, freqs[-1])
    plt.plot(freqs, pgram[91143], label='{:s} ({:s})'.format(id2word[91143], 'airplane'))
    plt.hlines(dps[91143], 0.0, df[91143])

    plt.scatter(df[91143], dps[91143], s=60, c='r')

    plt.xlabel('Frequency')
    plt.ylabel('Periodogram')
    plt.legend()
    plt.tight_layout()

    fig.set_size_inches(cm2inch(14, 8))
    fig.savefig(os.path.join(dirname, '{:d}_periodogram.eps'.format(91143)), format='eps', dpi=1200, bbox_inches='tight')
    plt.close(fig)
    plt.clf()

    print('Plotting finished')


def plot_aperiodic_features(feature_trajectories, dps, dp, dps_boundary, stream_length, id2word,
                            dirname='../aperiodic'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    aperiodic_features_indices = np.where((dps > dps_boundary) & (dp > np.floor(stream_length / 2)))[0]
    print('Plotting %d aperiodic features.' % len(aperiodic_features_indices))

    for i in aperiodic_features_indices:
        fig = plt.figure()
        fig.suptitle(id2word[i], fontsize=14)

        plot_aperiodic_column(feature_trajectories[i], 3, 1)
        plot_aperiodic_column(feature_trajectories[i], 7, 2)

        fig.set_size_inches(16, 10)
        plt.tight_layout()
        fig.savefig(os.path.join(dirname, '%s_%d.png' % (id2word[i], i)))
        plt.close(fig)
        plt.clf()

    print('Finished plotting aperiodic features.')


def plot_periodic_features(feature_trajectories, dps, dp, dps_boundary, stream_length, id2word, dirname='../periodic'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    periodic_features_indices = np.where((dps > dps_boundary) & (dp <= np.floor(stream_length / 2)))[0]
    print('Plotting %d periodic features.' % len(periodic_features_indices))

    for i in periodic_features_indices:
        fig = plt.figure()
        fig.suptitle(id2word[i], fontsize=14)

        plot_periodic_column(feature_trajectories[i], dp[i], 3, 1)
        plot_periodic_column(feature_trajectories[i], dp[i], 7, 2)

        fig.set_size_inches(16, 10)
        plt.tight_layout()
        fig.savefig(os.path.join(dirname, '%s_%d.png' % (id2word[i], i)))
        plt.close(fig)
        plt.clf()

    print('Finished plotting periodic features.')


def plot_aperiodic_column(feature_trajectory, window, col_num):
    n_days = len(feature_trajectory)
    days = np.arange(n_days)
    x = np.linspace(0.0, n_days, 1000)

    ma = moving_average(feature_trajectory, window)
    ma_mean = np.mean(ma)
    ma_std = np.std(ma)
    cutoff = ma_mean + ma_std

    # First graph: feature trajectory & moving average.
    plt.subplot(3, 2, col_num)
    plt.title('Window size: %d' % window)
    plt.xlim(0.0, n_days)
    plt.plot(days, feature_trajectory)
    plt.plot(days[window - 1:], ma, 'r', linewidth=2)
    plt.hlines(ma_mean, 0, n_days, colors='c')
    plt.hlines(cutoff, 0, n_days, colors='g', linestyles='dashed', linewidth=1.5)

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()

    # Second graph: Gaussian model using EM algorithm.
    g = gmm.GaussianMixture(covariance_type='diag')
    observations = np.hstack((days.reshape(-1, 1), feature_trajectory.reshape(-1, 1)))
    observations = observations[observations[:, 1] > cutoff, :]
    g.fit(observations)

    mean = g.means_[0, 0]
    std = np.sqrt(g.covariances_[0, 0])
    pdf = norm.pdf(x, mean, std)
    xticks_pos = [mean - std, mean, mean + std]
    xticks = [('%.2f' % (mean - std)), ('%.2f' % mean), ('%.2f' % (mean + std))]

    plt.subplot(3, 2, col_num + 2)
    plt.title('Gaussian model (EM algorithm)')
    plt.xlim(0.0, n_days)
    plt.plot(days, feature_trajectory)
    plt.plot(x, pdf, 'k', linewidth=1.5)

    plt.vlines(mean, 0.0, np.max(pdf), 'r')
    plt.axvspan(mean - std, mean + std, facecolor='g', alpha=0.5)
    plt.hlines(cutoff, 0, n_days, colors='g', linestyles='dashed', linewidth=1.5)
    plt.xticks(xticks_pos, xticks, rotation=45)

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()

    # Third graph: Gaussian model using least squares.
    def gaussian_curve(value, loc, scale):
        return norm.pdf(value, loc=loc, scale=scale)

    peak_indices = np.where(feature_trajectory > cutoff)
    peak_days = peak_indices[0]
    peaks = feature_trajectory[peak_indices].reshape(-1)
    peaks /= np.sum(peaks)  # Normalize the DFIDF so it can be interpreted as probability.

    p0 = (peak_days[len(peak_days) // 2], len(peak_days) / 4)
    popt, pcov = curve_fit(gaussian_curve, peak_days, peaks, p0=p0, bounds=(0.0, n_days))

    mean, std = popt
    pdf = gaussian_curve(x, mean, std)
    xticks_pos = [mean - std, mean, mean + std]
    xticks = [('%.2f' % (mean - std)), ('%.2f' % mean), ('%.2f' % (mean + std))]

    plt.subplot(3, 2, col_num + 4)
    plt.title('Gaussian model (Least squares)')
    plt.xlim(0.0, n_days)
    plt.plot(days, feature_trajectory)
    plt.plot(x, pdf, 'k', linewidth=1.5)

    plt.vlines(mean, 0.0, np.max(pdf), 'r')
    plt.axvspan(mean - std, mean + std, facecolor='g', alpha=0.5)
    plt.hlines(cutoff, 0, n_days, colors='g', linestyles='dashed', linewidth=1.5)
    plt.xticks(xticks_pos, xticks, rotation=45)

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()


def plot_periodic_column(feature_trajectory, dominant_period, window, col_num):
    n_days = len(feature_trajectory)
    days = np.arange(n_days)
    x = np.linspace(0.0, n_days, 1000)

    ma = moving_average(feature_trajectory.reshape(-1), window)
    ma_mean = np.mean(ma)
    ma_std = np.std(ma)
    cutoff = ma_mean + 0.5 * ma_std

    observations = np.hstack((days.reshape(-1, 1), feature_trajectory.reshape(-1, 1)))
    observations = observations[observations[:, 1] > cutoff, :]
    normalized_trajectory = feature_trajectory / np.sum(feature_trajectory)

    n_components = int(min(np.floor(n_days / dominant_period), len(observations)))
    g = gmm.GaussianMixture(n_components=n_components, covariance_type='diag')
    g.fit(observations)

    components = np.squeeze(np.array(
        [cauchy.pdf(x, mean[0], np.sqrt(2 * np.log(2)) * np.sqrt(cov[0])) for mean, cov in
         zip(g.means_, g.covariances_)]))

    pdf = g.weights_ @ components
    xticks_pos = [mean[0] for mean in g.means_]
    xticks = [('%.2f' % mean[0]) for mean in g.means_]

    # First graph: feature trajectory & moving average.
    plt.subplot(3, 2, col_num)
    plt.title('Window size: %d' % window)
    plt.xlim(0.0, n_days)
    plt.plot(days, feature_trajectory)
    plt.plot(days[window - 1:], ma, 'r', linewidth=2)
    plt.hlines(ma_mean, 0, n_days, colors='c')
    plt.hlines(cutoff, 0, n_days, colors='g', linestyles='dashed', linewidth=1.5)

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()

    # Second graph: Cauchy mixture model using EM algorithm.
    plt.subplot(3, 2, col_num + 2)
    plt.title('Cauchy mixture model (EM algorithm)')
    plt.xlim(0.0, n_days)
    plt.plot(days, normalized_trajectory)
    plt.plot(x, pdf, 'k', linewidth=1.5)

    plt.hlines(cutoff / np.sum(feature_trajectory), 0, n_days, colors='g', linestyles='dashed', linewidth=1.5)
    plt.xticks(xticks_pos, xticks, rotation=45)

    for mean_, cov_ in zip(g.means_, g.covariances_):
        loc = mean_[0]
        hwhm = np.sqrt(2 * np.log(2)) * np.sqrt(cov_[0])
        plt.vlines(loc, 0.0, np.max(normalized_trajectory), 'r')
        plt.axvspan(loc - hwhm, loc + hwhm, facecolor='g', alpha=0.15)

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()

    # Third graph: Cauchy mixture model components.
    plt.subplot(3, 2, col_num + 4)
    plt.title('Cauchy mixture components (EM algorithm)')
    plt.xlim(0.0, n_days)
    plt.plot(days, normalized_trajectory)

    plt.xticks(xticks_pos, xticks, rotation=45)

    for mean_, cov_ in zip(g.means_, g.covariances_):
        loc = mean_[0]
        hwhm = np.sqrt(2 * np.log(2)) * np.sqrt(cov_[0])
        mixture_component = cauchy.pdf(x, loc, hwhm)
        plt.plot(x, mixture_component)
        plt.vlines(loc, 0.0, np.max(mixture_component), 'r')

    plt.xlabel('Days')
    plt.ylabel('DFIDF')
    plt.legend()
