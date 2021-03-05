from __future__ import print_function

import itertools
import json
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from .IID_losses import IID_loss
from .data import create_handwriting_dataloaders
from .eval_metrics import _hungarian_match, _original_match, _acc
from .transforms import sobel_process


def _clustering_get_data(config, net, dataloader, sobel=False, using_IR=False, get_soft=False, verbose=None):
    """
    Returns cuda tensors for flat preds and targets.
    """

    assert (not using_IR)  # sanity; IR used by segmentation only

    num_batches = len(dataloader)
    flat_targets_all = torch.zeros((num_batches * dataloader.batch_size), dtype=torch.int32).cuda()
    flat_predss_all = [
        torch.zeros((num_batches * config.batch_sz), dtype=torch.int32).cuda() for _ in xrange(config.num_sub_heads)
    ]

    if get_soft:
        soft_predss_all = [
            torch.zeros((num_batches * config.batch_sz, config.output_k), dtype=torch.float32).cuda()
            for _ in xrange(config.num_sub_heads)
        ]

    num_test = 0
    for b_i, batch in enumerate(dataloader):
        imgs = batch[0].cuda()

        if sobel:
            imgs = sobel_process(imgs, config.include_rgb, using_IR=using_IR)

        flat_targets = batch[1]

        with torch.no_grad():
            x_outs = net(imgs)

        assert (x_outs[0].shape[1] == config.output_k)
        assert (len(x_outs[0].shape) == 2)

        num_test_curr = flat_targets.shape[0]
        num_test += num_test_curr

        start_i = b_i * dataloader.batch_size
        for i in xrange(config.num_sub_heads):
            x_outs_curr = x_outs[i]
            flat_preds_curr = torch.argmax(x_outs_curr, dim=1)  # along output_k
            flat_predss_all[i][start_i:(start_i + num_test_curr)] = flat_preds_curr

            if get_soft:
                soft_predss_all[i][start_i:(start_i + num_test_curr), :] = x_outs_curr

        flat_targets_all[start_i:(start_i + num_test_curr)] = flat_targets.cuda()

    flat_predss_all = [flat_predss_all[i][:num_test] for i in xrange(config.num_sub_heads)]
    flat_targets_all = flat_targets_all[:num_test]

    if not get_soft:
        return flat_predss_all, flat_targets_all
    else:
        soft_predss_all = [soft_predss_all[i][:num_test] for i in xrange(config.num_sub_heads)]

        return flat_predss_all, flat_targets_all, soft_predss_all


def cluster_subheads_eval(config, net,
                          mapping_assignment_dataloader,
                          mapping_test_dataloader,
                          sobel,
                          using_IR=False,
                          get_data_fn=_clustering_get_data,
                          use_sub_head=None,
                          verbose=0):
    """
    Used by both clustering and segmentation.
    Returns metrics for test set.
    Get result from average accuracy of all sub_heads (mean and std).
    All matches are made from training data.
    Best head metric, which is order selective unlike mean/std, is taken from
    best head determined by training data (but metric computed on test data).

    ^ detail only matters for IID+/semisup where there's a train/test split.

    Option to choose best sub_head either based on loss (set use_head in main
    script), or eval. Former does not use labels for the selection at all and this
    has negligible impact on accuracy metric for our models.
    """

    all_matches, train_accs = _get_assignment_data_matches(net,
                                                           mapping_assignment_dataloader,
                                                           config,
                                                           sobel=sobel,
                                                           using_IR=using_IR,
                                                           get_data_fn=get_data_fn,
                                                           verbose=verbose)

    best_sub_head_eval = np.argmax(train_accs)
    if (config.num_sub_heads > 1) and (use_sub_head is not None):
        best_sub_head = use_sub_head
    else:
        best_sub_head = best_sub_head_eval

    if config.mode == "IID":
        assert (config.mapping_assignment_partitions == config.mapping_test_partitions)
        test_accs = train_accs
    elif config.mode == "IID+":
        flat_predss_all, flat_targets_all, = get_data_fn(config, net, mapping_test_dataloader, sobel=sobel,
                                                         using_IR=using_IR,
                                                         verbose=verbose)

        num_samples = flat_targets_all.shape[0]
        test_accs = np.zeros(config.num_sub_heads, dtype=np.float32)
        for i in xrange(config.num_sub_heads):
            reordered_preds = torch.zeros(num_samples, dtype=flat_predss_all[0].dtype).cuda()
            for pred_i, target_i in all_matches[i]:
                reordered_preds[flat_predss_all[i] == pred_i] = target_i
            test_acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose=0)

            test_accs[i] = test_acc
    else:
        assert False

    return {"test_accs": list(test_accs),
            "avg": np.mean(test_accs),
            "std": np.std(test_accs),
            "best": test_accs[best_sub_head],
            "worst": test_accs.min(),
            "best_train_sub_head": best_sub_head,  # from training data
            "best_train_sub_head_match": all_matches[best_sub_head],
            "train_accs": list(train_accs)}


def _get_assignment_data_matches(net, mapping_assignment_dataloader, config,
                                 sobel=False,
                                 using_IR=False,
                                 get_data_fn=None,
                                 just_matches=False,
                                 verbose=0):
    """
    Get all best matches per head based on train set i.e. mapping_assign, and mapping_assign accs.
    """

    if verbose:
        print("calling cluster eval direct (helper) %s" % datetime.now())
        sys.stdout.flush()

    flat_predss_all, flat_targets_all = \
        get_data_fn(config, net, mapping_assignment_dataloader, sobel=sobel, using_IR=using_IR, verbose=verbose)

    if verbose:
        print("getting data fn has completed %s" % datetime.now())
        print("flat_targets_all %s, flat_predss_all[0] %s" %
              (list(flat_targets_all.shape), list(flat_predss_all[0].shape)))
        sys.stdout.flush()

    num_test = flat_targets_all.shape[0]
    if verbose == 2:
        print("num_test: %d" % num_test)
        for c in xrange(config.gt_k):
            print("gt_k: %d count: %d" % (c, (flat_targets_all == c).sum()))

    assert (flat_predss_all[0].shape == flat_targets_all.shape)
    num_samples = flat_targets_all.shape[0]

    all_matches = []
    if not just_matches:
        all_accs = np.zeros(config.num_sub_heads, dtype=np.float32)

    for i in xrange(config.num_sub_heads):
        if verbose:
            print("starting head %d with eval mode %s, %s" % (i, config.eval_mode, datetime.now()))
            sys.stdout.flush()

        if config.eval_mode == "hung":
            match = _hungarian_match(flat_predss_all[i], flat_targets_all,
                                     preds_k=config.output_k,
                                     targets_k=config.gt_k)
        elif config.eval_mode == "orig":
            match = _original_match(flat_predss_all[i], flat_targets_all,
                                    preds_k=config.output_k,
                                    targets_k=config.gt_k)
        else:
            assert False, "config mode should be one of [orig, hung]"

        if verbose:
            print("got match %s" % (datetime.now()))
            sys.stdout.flush()

        all_matches.append(match)

        if not just_matches:
            # reorder predictions to be same cluster assignments as gt_k
            found = torch.zeros(config.output_k)
            reordered_preds = torch.zeros(num_samples, dtype=flat_predss_all[0].dtype).cuda()

            for pred_i, target_i in match:
                reordered_preds[flat_predss_all[i] == pred_i] = target_i
                found[pred_i] = 1
                if verbose == 2:
                    print((pred_i, target_i))
            assert (found.sum() == config.output_k)  # each output_k must get mapped

            if verbose:
                print("reordered %s" % (datetime.now()))
                sys.stdout.flush()

            acc = _acc(reordered_preds, flat_targets_all, config.gt_k, verbose)
            all_accs[i] = acc

    if just_matches:
        return all_matches
    else:
        return all_matches, all_accs


def get_subhead_using_loss(config, dataloaders_head_B, net, sobel, lamb, compare=False):
    net.eval()

    head = "B"  # main output head
    dataloaders = dataloaders_head_B
    iterators = (d for d in dataloaders)

    b_i = 0
    loss_per_sub_head = np.zeros(config.num_sub_heads)
    for tup in itertools.izip(*iterators):
        net.module.zero_grad()

        dim = config.in_channels
        if sobel:
            dim -= 1

        all_imgs = torch.zeros(config.batch_sz, dim, config.input_sz[0], config.input_sz[1]).cuda()
        all_imgs_tf = torch.zeros(config.batch_sz, dim, config.input_sz[0], config.input_sz[1]).cuda()

        imgs_curr = tup[0][0]  # always the first
        curr_batch_sz = imgs_curr.size(0)
        for d_i in xrange(config.num_dataloaders):
            imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
            assert (curr_batch_sz == imgs_tf_curr.size(0))

            actual_batch_start = d_i * curr_batch_sz
            actual_batch_end = actual_batch_start + curr_batch_sz
            all_imgs[actual_batch_start:actual_batch_end, :, :, :] = imgs_curr.cuda()
            all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = imgs_tf_curr.cuda()

        curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
        all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
        all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

        if sobel:
            all_imgs = sobel_process(all_imgs, config.include_rgb)
            all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

        with torch.no_grad():
            x_outs = net(all_imgs, head=head)
            x_tf_outs = net(all_imgs_tf, head=head)

        for i in xrange(config.num_sub_heads):
            loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i], lamb=lamb)
            loss_per_sub_head[i] += loss.item()

        if b_i % 100 == 0:
            print("at batch %d" % b_i)
            sys.stdout.flush()
        b_i += 1

    best_sub_head_loss = np.argmin(loss_per_sub_head)

    if compare:
        print(loss_per_sub_head)
        print("best sub_head by loss: %d" % best_sub_head_loss)

        best_epoch = np.argmax(np.array(config.epoch_acc))
        if "best_train_sub_head" in config.epoch_stats[best_epoch]:
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_train_sub_head"]
            test_accs = config.epoch_stats[best_epoch]["test_accs"]
        else:  # older config version
            best_sub_head_eval = config.epoch_stats[best_epoch]["best_head"]
            test_accs = config.epoch_stats[best_epoch]["all"]

        print("best sub_head by eval: %d" % best_sub_head_eval)

        print("... loss select acc: %f, eval select acc: %f" %
              (test_accs[best_sub_head_loss],
               test_accs[best_sub_head_eval]))

    net.train()

    return best_sub_head_loss


def cluster_eval(config, net, mapping_assignment_dataloader,
                 mapping_test_dataloader, sobel,
                 use_sub_head=None, print_stats=False):
    if config.double_eval:
        # Pytorch's behaviour varies depending on whether .eval() is called or not
        # The effect is batchnorm updates if .eval() is not called
        # So double eval can be used (optionally) for IID, where train = test set.
        # https://pytorch.org/docs/master/nn.html#torch.nn.BatchNorm2d

        stats_dict2 = cluster_subheads_eval(config, net,
                                            mapping_assignment_dataloader=mapping_assignment_dataloader,
                                            mapping_test_dataloader=mapping_test_dataloader,
                                            sobel=sobel,
                                            use_sub_head=use_sub_head)

        if print_stats:
            print("double eval stats:")
            print(stats_dict2)
        else:
            config.double_eval_stats.append(stats_dict2)
            config.double_eval_acc.append(stats_dict2["best"])
            config.double_eval_avg_subhead_acc.append(stats_dict2["avg"])

    net.eval()
    stats_dict = cluster_subheads_eval(config, net,
                                       mapping_assignment_dataloader=mapping_assignment_dataloader,
                                       mapping_test_dataloader=mapping_test_dataloader,
                                       sobel=sobel,
                                       use_sub_head=use_sub_head)
    net.train()

    if print_stats:
        print("eval stats:")
        print(stats_dict)
    else:
        acc = stats_dict["best"]
        is_best = (len(config.epoch_acc) > 0) and (acc > max(config.epoch_acc))

        config.epoch_stats.append(stats_dict)
        config.epoch_acc.append(acc)
        config.epoch_avg_subhead_acc.append(stats_dict["avg"])

        return is_best


def get_subhead_cluster_stats(config, net):
    net.eval()

    # TODO: magic strings
    train_json_path = os.path.join("train", config.dataset + "_train.json")
    test_json_path = os.path.join("test", config.dataset + "_test.json")
    val_json_path = os.path.join("val", config.dataset + "_val.json")
    unlabelled_json_path = os.path.join("unlabelled", config.dataset + "_unlabelled_labelled.json")
    dataloader_list, mapping_assignment_dl, _ = create_handwriting_dataloaders(config, train_json_path, val_json_path,
                                                                               test_json_path, unlabelled_json_path,
                                                                               twohead=False)
    print("Dataloaders created")

    _, train_accs = _get_assignment_data_matches(net, mapping_assignment_dl, config, sobel=config.sobel,
                                                 using_IR=False, get_data_fn=_clustering_get_data, verbose=0)
    best_sub_head = np.argmax(train_accs)
    print("Best subhead determined")

    assert len(dataloader_list) == 1
    dataloader = dataloader_list[0][0]  # should be fine since all dataloaders contain the same data
    classes = dataloader.dataset.get_classes()
    flat_predss_all, flat_targets_all = _clustering_get_data(config, net, dataloader, sobel=config.sobel,
                                                             using_IR=False, verbose=0)
    print("Data predicted")

    num_samples = len(flat_targets_all)
    assert all([len(p) == num_samples for p in flat_predss_all])
    subhead_cluster_stats = []
    for sh_idx, subhead_prediction in enumerate(flat_predss_all):
        print("Processing subhead", sh_idx)
        cluster_stats = {}
        for i in range(num_samples):
            cluster_id = subhead_prediction[i].item()
            target = classes[flat_targets_all[i].item()]
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = []
            cluster_stats[cluster_id].append(target)
        cluster_stats = {k: Counter(v) for k, v in cluster_stats.items()}
        subhead_cluster_stats.append(cluster_stats)

    with open("cluster_stats.json", "") as out_f:
        json.dump({
            "best_sub_head": best_sub_head,
            "subhead_cluster_stats": subhead_cluster_stats
        }, out_f)

    return subhead_cluster_stats, best_sub_head


def plot_cluster_dist_per_class(config, subhead_cluster_stats):
    # rearrange data structure
    print("Plotting cluster dist per class")
    num_subheads = config.num_sub_heads
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
            relative_y = [float(s) / sum(y) for s in y]
            ax.set_title(class_name)
            ax.bar(x, relative_y)
    fig.tight_layout(pad=3.0, h_pad=4.0)
    plt.savefig("cluster_dist_per_class.png")


def plot_aligned_clusters(config, subhead_cluster_stats):
    # Number of predicted samples per cluster
    print("Plotting aligned clusters")
    plt.clf()
    matplotlib.rcParams.update({'font.size': 22})
    num_subheads = config.num_sub_heads
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
    plt.savefig("cluster_bars_aligned.png")


def plot_unaligned_clusters(config, subhead_cluster_stats):
    # working code for unaligned clusters
    print("Plotting unaligned clusters")
    plt.clf()
    matplotlib.rcParams.update({'font.size': 22})
    max_num_clusters = max([len(s) for s in subhead_cluster_stats])
    num_subheads = config.num_sub_heads
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
    plt.savefig("cluster_bars_unaligned.png")


def plot_cluster_stats(config, net):
    # TODO
    #   - plot original data dists
    #   - for each subhead:
    #       - analyse individual clusters and create class distribution (rank them based on purity)
    #       - maybe plot PCA of individual clusters separately
    #       - plot PCA of all clusters in single plot - might be messy because there are 35 clusters
    #   - double check data gathering

    subhead_cluster_stats, best_sub_head = get_subhead_cluster_stats(config, net)
    # with open("cluster_stats.json") as csf:
    #     subhead_cluster_stats = json.load(csf)

    plot_cluster_dist_per_class(config, subhead_cluster_stats)
    plot_aligned_clusters(config, subhead_cluster_stats)
    plot_unaligned_clusters(config, subhead_cluster_stats)
