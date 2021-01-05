import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable

from src.utils.cluster.transforms import sobel_process


def get_dlen(net_features, dataloader, include_rgb=None, penultimate_features=False):
    imgs, _ = next(iter(dataloader))
    img = torch.unsqueeze(imgs[0], 0)
    sobel_img = sobel_process(img.cuda(), include_rgb)

    net_features.eval()
    with torch.no_grad():
        x_features = net_features(sobel_img, trunk_features=True, penultimate_features=penultimate_features)
    net_features.train()

    dlen = x_features.view(x_features.shape[0], -1).shape[1]

    return dlen


def assess_acc(net, test_loader, gt_k=None, include_rgb=None, penultimate_features=False):
    correct = 0
    total = 0
    for i, (imgs, targets) in enumerate(test_loader):
        imgs = sobel_process(imgs.cuda(), include_rgb)

        with torch.no_grad():
            x_out = net(imgs, penultimate_features=penultimate_features)

        # bug fix!!
        preds = np.argmax(x_out.cpu().numpy(), axis=1).astype("int")
        targets = targets.numpy().astype("int")
        assert (preds.min() >= 0 and preds.max() < gt_k)
        assert (targets.min() >= 0 and targets.max() < gt_k)
        assert (preds.shape == targets.shape)

        correct += (preds == targets).sum()
        total += preds.shape[0]

    return correct / float(total)


def assess_acc_block(net, test_loader, gt_k=None, include_rgb=None, penultimate_features=False, contiguous_sz=None):
    total = 0
    all_outputs = None
    all_targets = None
    for i, (imgs, targets) in enumerate(test_loader):
        imgs = sobel_process(imgs.cuda(), include_rgb)

        with torch.no_grad():
            x_out = net(imgs, penultimate_features=penultimate_features)

        bn, dlen = x_out.shape
        if all_outputs is None:
            all_outputs = np.zeros((len(test_loader) * bn, dlen))
            all_targets = np.zeros(len(test_loader) * bn)

        all_outputs[total:(total + bn), :] = x_out.cpu().numpy()
        all_targets[total:(total + bn)] = targets.numpy()
        total += bn

    # 40000
    all_outputs = all_outputs[:total, :]
    all_targets = all_targets[:total]

    num_orig, leftover = divmod(total, contiguous_sz)
    assert (leftover == 0)

    all_outputs = all_outputs.reshape((num_orig, contiguous_sz, dlen))
    all_outputs = all_outputs.sum(axis=1, keepdims=False) / float(contiguous_sz)

    all_targets = all_targets.reshape((num_orig, contiguous_sz))
    # sanity check
    all_targets_avg = all_targets.astype("int").sum(axis=1) / contiguous_sz
    all_targets = all_targets[:, 0].astype("int")
    assert (np.array_equal(all_targets_avg, all_targets))

    preds = np.argmax(all_outputs, axis=1).astype("int")
    assert (preds.min() >= 0 and preds.max() < gt_k)
    assert (all_targets.min() >= 0 and all_targets.max() < gt_k)
    if not (preds.shape == all_targets.shape):
        print(preds.shape)
        print(all_targets.shape)
        assert False

    assert (preds.shape == (num_orig,))
    correct = (preds == all_targets).sum()

    return correct / float(num_orig)


def ensure_all_batchnorm_track(net):
    print("ensure_all_batchnorm_track:")
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if not (m.track_running_stats):
                print("... setting non-track batchnorm to track stats")
                m.track_running_stats = True
