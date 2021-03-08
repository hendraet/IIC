import os

import matplotlib
import matplotlib.pyplot as plt
import numpy
from matplotlib import cm
from sklearn.decomposition import PCA
from tqdm import tqdm


def highlight_best_subhead(best_subhead, fig, axs):
    best_subhead_axis = axs[best_subhead, :]
    for ax in best_subhead_axis:
        bbox = ax.get_position()
        rect = matplotlib.patches.Rectangle((0, bbox.y0), 1, bbox.height, color="#32cd3205", zorder=-1,
                                            transform=fig.transFigure, clip_on=False)
        ax.add_artist(rect)
    for ax in best_subhead_axis.flat:
        ax.patch.set_visible(False)


def plot_cluster_dist_per_class(config, subhead_cluster_stats, best_subhead):
    # rearrange data structure
    print("Plotting cluster dist per class")
    num_subheads = config.num_subheads
    permuted_subhead_cluster_stats = []
    for subhead_stats in subhead_cluster_stats:
        class_cluster_mapping = {}
        for cluster_id in subhead_stats:
            for class_name, num_samples in subhead_stats[cluster_id].items():
                if class_name not in class_cluster_mapping:
                    class_cluster_mapping[class_name] = {}
                class_cluster_mapping[class_name][cluster_id] = num_samples
        permuted_subhead_cluster_stats.append(class_cluster_mapping)

    # plotting
    matplotlib.rcParams.update({'font.size': 22})
    max_num_clusters = config.output_ks[0]
    plt.clf()
    num_classes = max(len(d) for d in permuted_subhead_cluster_stats)  # TODO: might not work in all cases
    fig, axs = plt.subplots(num_subheads, num_classes, sharey="all", figsize=(num_classes * 7, num_subheads * 5))
    for sh_idx, subhead_stats in enumerate(permuted_subhead_cluster_stats):
        for class_name, ax in zip(sorted(subhead_stats.keys()), axs[sh_idx]):
            class_stats = subhead_stats[class_name]
            x = [i for i in range(max_num_clusters)]
            y = [class_stats[str(i)] if str(i) in class_stats else 0.0 for i in range(max_num_clusters)]
            relative_y = [float(s) / sum(y) for s in y]  # TODO: check why there can be a division by 0
            ax.set_title(class_name)
            ax.bar(x, relative_y)

    fig.tight_layout(pad=3.0, h_pad=4.0)
    highlight_best_subhead(best_subhead, fig, axs)
    plt.savefig(os.path.join(config.result_dir, "cluster_dist_per_class.png"))


def plot_aligned_clusters(config, subhead_cluster_stats, best_subhead):
    # Number of predicted samples per cluster
    print("Plotting aligned clusters")
    plt.clf()
    matplotlib.rcParams.update({'font.size': 22})
    num_subheads = config.num_subheads
    max_num_clusters = config.output_ks[0]
    fig, axs = plt.subplots(num_subheads, max_num_clusters, sharex="all", sharey="all",
                            figsize=(max_num_clusters * 4, num_subheads * 5))
    for sh_idx, subhead_stats in enumerate(subhead_cluster_stats):
        for cluster_id in sorted(subhead_stats.keys(), key=int):
            items = list(subhead_stats[cluster_id].items())
            x, y = zip(*sorted(items, key=lambda x: x[0]))
            relative_y = [float(s) / sum(y) for s in y]

            ax = axs[sh_idx][int(cluster_id)]
            ax.bar(x, relative_y)
        for i, ax in enumerate(axs.flat):
            ax.set_title(i % max_num_clusters)
        for ax in axs[-1, :]:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    fig.tight_layout(pad=1.5)
    highlight_best_subhead(best_subhead, fig, axs)
    plt.savefig(os.path.join(config.result_dir, "cluster_bars_aligned.png"))


def plot_unaligned_clusters(config, subhead_cluster_stats, best_subhead):
    # working code for unaligned clusters
    print("Plotting unaligned clusters")
    plt.clf()
    matplotlib.rcParams.update({'font.size': 22})
    max_num_clusters = max([len(s) for s in subhead_cluster_stats])
    num_subheads = config.num_subheads
    fig, axs = plt.subplots(num_subheads, max_num_clusters, sharex="all", sharey="all",
                            figsize=(max_num_clusters * 3, num_subheads * 5))
    for sh_idx, subhead_stats in enumerate(subhead_cluster_stats):
        for i, cluster_id in enumerate(sorted(subhead_stats.keys(), key=int)):
            items = list(subhead_stats[cluster_id].items())
            x, y = zip(*sorted(items, key=lambda x: x[0]))
            relative_y = [float(s) / sum(y) for s in y]

            ax = axs[sh_idx][i]
            ax.set_title(cluster_id)
            ax.bar(x, relative_y)
        for ax in axs[-1, :]:
            for tick in ax.get_xticklabels():
                tick.set_rotation(90)
    fig.tight_layout(pad=3.0, h_pad=4.0)
    highlight_best_subhead(best_subhead, fig, axs)
    plt.savefig(os.path.join(config.result_dir, "cluster_bars_unaligned.png"))


def plot_clusters(config, embeddings, sample_info, labelled_classes=()):
    labels = [sample["type"] for sample in sample_info]
    possible_labels = set(labels)
    num_possible_labels = len(possible_labels)
    predicted_clusters = set([sample["prediction"] for sample in sample_info])
    num_clusters = len(predicted_clusters)
    num_rows = int(numpy.sqrt(num_clusters))
    num_cols = int(numpy.ceil(num_clusters / num_rows))

    plt.clf()
    subplot_size = 10
    # Oh yeah, matplotlib again...
    # Args are rows, columns but figsize is width x height. Totally logical
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * subplot_size, num_rows * subplot_size))
    num_labelled_classes = len(labelled_classes)
    if num_labelled_classes > 0:
        assert num_labelled_classes <= num_possible_labels
        labelled_colours = cm.autumn(numpy.linspace(0, 1, num_labelled_classes))
        labelled_colour_map = {label: colour for label, colour in zip(labelled_classes, labelled_colours)}
        unlabelled_colours = cm.winter(numpy.linspace(0, 1, num_possible_labels - num_labelled_classes))
        unlabelled_classes = possible_labels - set(labelled_classes)
        unlabelled_colour_map = {label: colour for label, colour in zip(unlabelled_classes, unlabelled_colours)}

        label_colour_map = unlabelled_colour_map.copy()
        label_colour_map.update(labelled_colour_map)
    else:
        if num_possible_labels <= 10:
            colour_palette = iter([plt.cm.tab10(i) for i in range(num_possible_labels)])
        elif num_possible_labels <= 20:
            colour_palette = iter([plt.cm.tab20(i) for i in range(num_possible_labels)])
        else:
            colour_palette = cm.rainbow(numpy.linspace(0, 1, num_possible_labels))
        label_colour_map = {label: colour for label, colour in zip(possible_labels, colour_palette)}

    print("Plotting PCA for each cluster...")
    for cluster, ax in tqdm(zip(predicted_clusters, axs.flat)):
        relevant_indices = [idx for idx, sample in enumerate(sample_info) if sample["prediction"] == cluster]
        if len(relevant_indices) < 2:
            print("too few samples for cluster " + str(cluster))
            continue
        relevant_embeddings = numpy.take(numpy.array(embeddings), relevant_indices, axis=0)
        relevant_labels = numpy.take(numpy.asarray(labels), relevant_indices)

        pca = PCA(n_components=2)
        fitted_data = pca.fit_transform(relevant_embeddings)
        x = fitted_data[:, 0]
        y = fitted_data[:, 1]

        ax.set_title(str(cluster))
        # gather all the indices of one subclass and plot them class by class, so that they are correctly coloured
        # and named
        for label in possible_labels:
            indices = []
            for idx, relevant_label in enumerate(relevant_labels):
                if label == relevant_label:
                    indices.append(idx)
            if len(indices) > 0:
                coords = numpy.asarray([[x[idx], y[idx]] for idx in indices])
                ax.scatter(coords[:, 0], coords[:, 1], c=label_colour_map[label], label=label)
        ax.legend()

    fig.tight_layout() # (pad=3.0, h_pad=4.0)
    if num_labelled_classes == 0:
        plt.savefig(os.path.join(config.result_dir, "pca_per_cluster.png"))
    else:
        plt.savefig(os.path.join(config.result_dir, "pca_per_cluster_binary.png"))

    # # gather all the indices of one subclass and plot them class by class, so that they are correctly coloured and named
    # label_set = set(labels)
    # for item in label_set:
    #     indices = []
    #     for idx, label in enumerate(labels):
    #         if item == label:
    #             indices.append(idx)
    #     if len(indices) > 0:
    #         coords = numpy.asarray([[x[idx], y[idx]] for idx in indices])
    #         ax.scatter(coords[:, 0], coords[:, 1], label=item)
