import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.mixture as gmm
import wordcloud
from scipy.optimize import curve_fit
from scipy.stats import cauchy, norm

import event_detection.postprocessing as post
from event_detection.original_method import moving_average

matplotlib.rc('font', family='DejaVu Sans')


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


def plot_events(feature_trajectories, events, id2word, dps, dp, dirname='../events'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    n_days = feature_trajectories.shape[1]
    days = np.arange(feature_trajectories.shape[1])

    for i, event in enumerate(events):
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

        event_trajectory, event_period = post.create_event_trajectory(event, feature_trajectories, dps, dp)
        plt.plot(days, event_trajectory, label='Period: {:d}'.format(event_period), color='red', linewidth=1.5)

        if event_period == n_days:
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
