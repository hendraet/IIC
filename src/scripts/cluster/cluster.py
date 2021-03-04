from __future__ import print_function

import itertools
import os
import pickle
import sys
from datetime import datetime
from logging import warn

import matplotlib
import numpy as np
import torch

from src.utils.cluster.render import save_progress
from src.utils.utils import get_std_arg_parser

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import src.archs as archs
from src.utils.cluster.general import config_to_str, get_opt, update_lr, nice
from src.utils.cluster.transforms import sobel_process
from src.utils.cluster.cluster_eval import cluster_eval, get_subhead_using_loss, evaluate_clusters
from src.utils.cluster.data import get_dataloader_list
from src.utils.cluster.IID_losses import IID_loss

"""
  Fully unsupervised clustering ("IIC" = "IID").
  Train and test script (coloured datasets).
  Network has two heads, for overclustering and final clustering.
"""


def parse_config():
    parser = get_std_arg_parser()
    parser.add_argument("dataset_root", type=str, help="dir where the dataset dir is located")
    parser.add_argument("dataset", type=str)
    parser.add_argument("--opt", type=str, default="Adam")
    parser.add_argument("--mode", type=str)
    parser.add_argument("--sobel", default=False, action="store_true")

    parser.add_argument("--gt_k", type=int, default=5, help="actual number of classes in the dataset")
    parser.add_argument("--output_ks", type=int, nargs="+",
                        help="number of classes that each head should produce - can be used for overclustering")

    parser.add_argument("--lamb", type=float, default=1.0)

    parser.add_argument("--batch_sz", type=int, required=True)  # num pairs
    parser.add_argument("--num_dataloaders", type=int, default=3)
    parser.add_argument("--num_sub_heads", type=int, default=5)  # per head...

    parser.add_argument("--leave_out_unlabelled", default=False, action="store_true")

    parser.add_argument("--save_freq", type=int, default=20)

    parser.add_argument("--double_eval", default=False, action="store_true",
                        help="Adds an additional evaluation step that evaluates the model in train mode instead of eval"
                             " mode")

    parser.add_argument("--reverse_heads", default=False, action="store_true")
    parser.add_argument("--head_epochs", type=int, nargs="+", default=[1])

    parser.add_argument("--batchnorm_track", default=False, action="store_true",
                        help="tracks batchnorm running stats")
    parser.add_argument("--save_progression", default=False, action="store_true")
    parser.add_argument("--select_sub_head_on_loss", default=False, action="store_true")  # TODO?
    parser.add_argument("--evaluate_clusters", default=False, action="store_true")

    # transforms
    parser.add_argument("--mix_train", dest="mix_train", default=False, action="store_true",
                        help="argument for STL10 dataset where unlabelled data is added to the train set")
    parser.add_argument("--include_rgb", dest="include_rgb", default=False, action="store_true")

    parser.add_argument("--demean", dest="demean", default=False, action="store_true")
    parser.add_argument("--per_img_demean", dest="per_img_demean", default=False, action="store_true")
    parser.add_argument("--data_mean", type=float, nargs="+", default=[0.5, 0.5, 0.5])
    parser.add_argument("--data_std", type=float, nargs="+", default=[0.5, 0.5, 0.5])

    parser.add_argument("--crop_orig", dest="crop_orig", default=False, action="store_true")
    parser.add_argument("--crop_other", dest="crop_other", default=False, action="store_true")
    parser.add_argument("--tf1_crop", type=str, default="random")  # type name
    parser.add_argument("--tf2_crop", type=str, default="random")
    parser.add_argument("--tf1_crop_sz", type=float, default=0.9, help="factor of how much smaller the crop should be")
    parser.add_argument("--tf2_crop_szs", type=float, nargs="+", default=[0.8, 0.9, 1.0])  # allow diff crop for imgs_tf
    parser.add_argument("--tf3_crop_diff", dest="tf3_crop_diff", default=False, action="store_true")
    parser.add_argument("--tf3_crop_sz", type=int, default=0)
    parser.add_argument("--input_sz", type=int, nargs="+", default=[64, 216])

    parser.add_argument("--fluid_warp", dest="fluid_warp", default=False, action="store_true")
    parser.add_argument("--rand_crop_sz", type=float, default=0.9)
    parser.add_argument("--rand_crop_szs_tf", type=float, nargs="+",
                        default=[0.8, 0.9, 1.0])  # only used if fluid warp true
    parser.add_argument("--rot_val", type=float, default=0.)  # only used if fluid warp true

    parser.add_argument("--always_rot", dest="always_rot", default=False, action="store_true")
    parser.add_argument("--no_jitter", dest="no_jitter", default=False, action="store_true")
    parser.add_argument("--no_flip", dest="no_flip", default=False, action="store_true")

    parser.add_argument("--cutout", default=False, action="store_true")
    parser.add_argument("--cutout_p", type=float, default=0.5)
    parser.add_argument("--cutout_max_box", type=float, default=0.5)

    config = parser.parse_args()
    if len(config.input_sz) == 1:
        config.input_sz = config.input_sz + config.input_sz
    elif len(config.input_sz) > 2:
        config.input_sz = config.input_sz[:2]

    if config.evaluate_clusters:
        config.num_dataloaders = 1
        if not config.restart:
            warn("Evaluating clusters on untrained model. Use --restart to reuse a trained model")

    return config


def setup(config):
    if config.mode == "IID":
        assert ("TwoHead" in config.arch)
        # Exactly one config has to match the groundtruth k and all ks have to be bigger than gt_k
        assert any(k == config.gt_k for k in config.output_ks)
        assert all(k >= config.gt_k for k in config.output_ks)
        config.output_k = config.gt_k
        config.eval_mode = "hung"
        config.twohead = True
    elif config.mode == "IID+":
        assert len(config.output_ks) == 1 and config.output_ks[0] >= config.gt_k
        config.output_k = config.output_ks[0]
        config.eval_mode = "orig"
        config.twohead = False
        config.double_eval = False
    else:
        raise NotImplementedError

    if config.sobel:
        if not config.include_rgb:
            config.in_channels = 2
        else:
            config.in_channels = 5
    else:
        config.in_channels = 1
        config.train_partitions = [True, False]
        config.mapping_assignment_partitions = [True, False]
        config.mapping_test_partitions = [True, False]

    config.out_dir = os.path.join(config.out_root, str(config.model_ind))
    assert (config.batch_sz % config.num_dataloaders == 0)
    config.dataloader_batch_sz = config.batch_sz / config.num_dataloaders

    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    if config.restart:
        config_name = "config.pickle"
        net_name = "latest_net.pytorch"
        opt_name = "latest_optimiser.pytorch"

        if config.restart_from_best:
            config_name = "best_config.pickle"
            net_name = "best_net.pytorch"
            opt_name = "best_optimiser.pytorch"

        given_config = config
        reloaded_config_path = os.path.join(given_config.out_dir, config_name)
        print("Loading restarting config from: %s" % reloaded_config_path)
        with open(reloaded_config_path, "rb") as config_f:
            config = pickle.load(config_f)
        assert given_config.test_code or (config.model_ind == given_config.model_ind)
        config.restart = True
        config.restart_from_best = given_config.restart_from_best

        if given_config.dataset_root is not None:
            config.dataset_root = given_config.dataset_root
        if given_config.out_root is not None:
            config.out_root = given_config.out_root
            config.out_dir = os.path.join(given_config.out_root, str(given_config.model_ind))

        # copy over new num_epochs and lr schedule
        config.num_epochs = given_config.num_epochs
        config.lr_schedule = given_config.lr_schedule
        config.save_progression = given_config.save_progression

        if not hasattr(config, "cutout"):
            config.cutout = False
            config.cutout_p = 0.5
            config.cutout_max_box = 0.5

        if not hasattr(config, "batchnorm_track"):
            config.batchnorm_track = True  # before we added in false option

        config.evaluate_clusters = given_config.evaluate_clusters

    else:
        print("Config: %s" % config_to_str(config))
        net_name = None
        opt_name = None

    net = archs.__dict__[config.arch](config)
    if config.restart:
        model_path = os.path.join(config.out_dir, net_name)
        print("Model path: %s" % model_path)
        net.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

    net.cuda()
    net = torch.nn.DataParallel(net)
    net.train()

    optimiser = get_opt(config.opt)(net.module.parameters(), lr=config.lr)
    if config.restart:
        opt_path = os.path.join(config.out_dir, opt_name)
        optimiser.load_state_dict(torch.load(opt_path))

    return config, net, optimiser


# TODO: center crop ok or does it remove too much information if text is aligned left
def train(config, net, optimiser, render_count=-1):
    dataloader_list, mapping_assignment_dataloader, mapping_test_dataloader = get_dataloader_list(config)

    num_heads = len(config.output_ks)

    # Results ----------------------------------------------------------------------

    if config.restart:
        if not config.restart_from_best:
            next_epoch = config.last_epoch + 1  # corresponds to last saved model
        else:
            next_epoch = np.argmax(np.array(config.epoch_acc)) + 1
        print("starting from epoch %d" % next_epoch)

        # in case we overshot without saving
        config.epoch_acc = config.epoch_acc[:next_epoch]  # in case we overshot
        config.epoch_avg_subhead_acc = config.epoch_avg_subhead_acc[:next_epoch]
        config.epoch_stats = config.epoch_stats[:next_epoch]

        if config.double_eval:
            config.double_eval_acc = config.double_eval_acc[:next_epoch]
            config.double_eval_avg_subhead_acc = config.double_eval_avg_subhead_acc[:next_epoch]
            config.double_eval_stats = config.double_eval_stats[:next_epoch]

        for i, loss in enumerate(config.epoch_loss):
            config.epoch_loss[i] = loss[:(next_epoch - 1)]
        for i, loss_no_lamb in enumerate(config.epoch_loss_no_lamb):
            config.epoch_loss_no_lamb[i] = loss_no_lamb[:(next_epoch - 1)]

    else:
        config.epoch_acc = []
        config.epoch_avg_subhead_acc = []
        config.epoch_stats = []

        if config.double_eval:
            config.double_eval_acc = []
            config.double_eval_avg_subhead_acc = []
            config.double_eval_stats = []

        config.epoch_loss = [[] for _ in range(num_heads)]
        config.epoch_loss_no_lamb = [[] for _ in range(num_heads)]

        sub_head = None
        if config.select_sub_head_on_loss:
            assert num_heads == 2
            sub_head = get_subhead_using_loss(config, dataloader_list[1], net, sobel=config.sobel, lamb=config.lamb)

        _ = cluster_eval(config, net,
                         mapping_assignment_dataloader=mapping_assignment_dataloader,
                         mapping_test_dataloader=mapping_test_dataloader,
                         sobel=config.sobel,
                         use_sub_head=sub_head)

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        if config.double_eval:
            print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()
        next_epoch = 1

    fig, axarr = plt.subplots(2 + 2 * num_heads + 2 * int(config.double_eval), sharex=False, figsize=(20, 20))

    save_progression = hasattr(config, "save_progression") and config.save_progression
    if save_progression:
        save_progression_count = 0
        save_progress(config, net, mapping_assignment_dataloader,
                      mapping_test_dataloader, save_progression_count,
                      sobel=config.sobel,
                      render_count=render_count)
        save_progression_count += 1

    # Train ------------------------------------------------------------------------

    heads = range(num_heads)
    if config.reverse_heads:
        heads = reversed(heads)

    for e_i in xrange(next_epoch, config.num_epochs + 1):
        print("Starting e_i: %d" % e_i)

        if e_i in config.lr_schedule:
            optimiser = update_lr(optimiser, lr_mult=config.lr_mult)

        for head_idx in heads:
            dataloaders = dataloader_list[head_idx]
            epoch_loss = config.epoch_loss[head_idx]
            epoch_loss_no_lamb = config.epoch_loss_no_lamb[head_idx]

            avg_loss = 0.  # over heads and head_epochs (and sub_heads)
            avg_loss_no_lamb = 0.
            avg_loss_count = 0

            for head_i_epoch in range(config.head_epochs[head_idx]):
                sys.stdout.flush()

                iterators = (d for d in dataloaders)

                b_i = 0
                for tup in itertools.izip(*iterators):
                    net.module.zero_grad()

                    in_channels = config.in_channels
                    if config.sobel:
                        # one less because this is before sobel
                        in_channels -= 1
                    all_imgs = torch.zeros((config.batch_sz, in_channels,
                                            config.input_sz[0], config.input_sz[1])).cuda()
                    all_imgs_tf = torch.zeros((config.batch_sz, in_channels,
                                               config.input_sz[0], config.input_sz[1])).cuda()

                    imgs_curr = tup[0][0]  # always the first
                    curr_batch_sz = imgs_curr.size(0)
                    for d_i in xrange(config.num_dataloaders):
                        imgs_tf_curr = tup[1 + d_i][0]  # from 2nd to last
                        assert (curr_batch_sz == imgs_tf_curr.size(0))

                        actual_batch_start = d_i * curr_batch_sz
                        actual_batch_end = actual_batch_start + curr_batch_sz
                        all_imgs[actual_batch_start:actual_batch_end, :, :, :] = imgs_curr.cuda()
                        all_imgs_tf[actual_batch_start:actual_batch_end, :, :, :] = imgs_tf_curr.cuda()

                    if not (curr_batch_sz == config.dataloader_batch_sz):
                        print("last batch sz %d" % curr_batch_sz)

                    curr_total_batch_sz = curr_batch_sz * config.num_dataloaders
                    all_imgs = all_imgs[:curr_total_batch_sz, :, :, :]
                    all_imgs_tf = all_imgs_tf[:curr_total_batch_sz, :, :, :]

                    if config.sobel:
                        all_imgs = sobel_process(all_imgs, config.include_rgb)
                        all_imgs_tf = sobel_process(all_imgs_tf, config.include_rgb)

                    x_outs = net(all_imgs, head_idx=head_idx)
                    x_tf_outs = net(all_imgs_tf, head_idx=head_idx)

                    avg_loss_batch = None  # avg over the sub_heads
                    avg_loss_no_lamb_batch = None
                    for i in xrange(config.num_sub_heads):
                        loss, loss_no_lamb = IID_loss(x_outs[i], x_tf_outs[i], lamb=config.lamb)
                        if avg_loss_batch is None:
                            avg_loss_batch = loss
                            avg_loss_no_lamb_batch = loss_no_lamb
                        else:
                            avg_loss_batch += loss
                            avg_loss_no_lamb_batch += loss_no_lamb

                    avg_loss_batch /= config.num_sub_heads
                    avg_loss_no_lamb_batch /= config.num_sub_heads

                    if ((b_i % 100) == 0) or (e_i == next_epoch and b_i < 10):
                        print("Model ind %d epoch %d head %s head_i_epoch %d batch %d: avg loss %f avg loss no lamb %f "
                              "time %s" % (config.model_ind, e_i, str(head_idx), head_i_epoch, b_i,
                                           avg_loss_batch.item(), avg_loss_no_lamb_batch.item(), datetime.now()))
                        sys.stdout.flush()

                    if not np.isfinite(avg_loss_batch.item()):
                        print("Loss is not finite... %s:" % str(avg_loss_batch))
                        sys.exit(1)

                    avg_loss += avg_loss_batch.item()
                    avg_loss_no_lamb += avg_loss_no_lamb_batch.item()
                    avg_loss_count += 1

                    avg_loss_batch.backward()
                    optimiser.step()

                    if ((b_i % 50) == 0) and save_progression:
                        save_progress(config, net, mapping_assignment_dataloader,
                                      mapping_test_dataloader, save_progression_count,
                                      sobel=False,
                                      render_count=render_count)
                        save_progression_count += 1

                    b_i += 1
                    if b_i == 2 and config.test_code:
                        break

            avg_loss = float(avg_loss / avg_loss_count)
            avg_loss_no_lamb = float(avg_loss_no_lamb / avg_loss_count)

            epoch_loss.append(avg_loss)
            epoch_loss_no_lamb.append(avg_loss_no_lamb)

        # Eval -----------------------------------------------------------------------

        # Can also pick the subhead using the evaluation process (to do this, set use_sub_head=None)
        sub_head = None
        if config.select_sub_head_on_loss:
            assert num_heads == 2
            sub_head = get_subhead_using_loss(config, dataloader_list[1], net, sobel=config.sobel, lamb=config.lamb)

        is_best = cluster_eval(config, net,
                               mapping_assignment_dataloader=mapping_assignment_dataloader,
                               mapping_test_dataloader=mapping_test_dataloader,
                               sobel=config.sobel,
                               use_sub_head=sub_head)

        print("Pre: time %s: \n %s" % (datetime.now(), nice(config.epoch_stats[-1])))
        if config.double_eval:
            print("double eval: \n %s" % (nice(config.double_eval_stats[-1])))
        sys.stdout.flush()

        axarr[0].clear()
        axarr[0].plot(config.epoch_acc)
        axarr[0].set_title("acc (best), top: %f" % max(config.epoch_acc))

        axarr[1].clear()
        axarr[1].plot(config.epoch_avg_subhead_acc)
        axarr[1].set_title("acc (avg), top: %f" % max(config.epoch_avg_subhead_acc))

        last_ax_idx = 1
        starting_ax_idx = last_ax_idx + 1
        for i in range(num_heads):
            axarr[starting_ax_idx + i].clear()
            axarr[starting_ax_idx + i].plot(config.epoch_loss[i])
            axarr[starting_ax_idx + i].set_title("Loss head_idx " + str(i))

            axarr[starting_ax_idx + i + 1].clear()
            axarr[starting_ax_idx + i + 1].plot(config.epoch_loss_no_lamb[i])
            axarr[starting_ax_idx + i + 1].set_title("Loss no lamb head_idx " + str(i))

        if config.double_eval:
            next_index = starting_ax_idx + 2 * num_heads
            axarr[next_index].clear()
            axarr[next_index].plot(config.double_eval_acc)
            axarr[next_index].set_title("double eval acc (best), top: %f" % max(config.double_eval_acc))

            axarr[next_index + 1].clear()
            axarr[next_index + 1].plot(config.double_eval_avg_subhead_acc)
            axarr[next_index + 1].set_title("double eval acc (avg), top: %f" % max(config.double_eval_avg_subhead_acc))

        fig.tight_layout()
        fig.canvas.draw_idle()
        fig.savefig(os.path.join(config.out_dir, "plots.png"))

        if is_best or (e_i % config.save_freq == 0):
            net.module.cpu()

            if e_i % config.save_freq == 0:
                torch.save(net.module.state_dict(), os.path.join(config.out_dir, "latest_net.pytorch"))
                torch.save(optimiser.state_dict(), os.path.join(config.out_dir, "latest_optimiser.pytorch"))
                config.last_epoch = e_i  # for last saved version

            if is_best:
                # also serves as backup if hardware fails - less likely to hit this
                torch.save(net.module.state_dict(), os.path.join(config.out_dir, "best_net.pytorch"))
                torch.save(optimiser.state_dict(), os.path.join(config.out_dir, "best_optimiser.pytorch"))

                with open(os.path.join(config.out_dir, "best_config.pickle"), 'wb') as outfile:
                    pickle.dump(config, outfile)

                with open(os.path.join(config.out_dir, "best_config.txt"), "w") as text_file:
                    text_file.write("%s" % config)

            net.module.cuda()

        with open(os.path.join(config.out_dir, "config.pickle"), 'wb') as outfile:
            pickle.dump(config, outfile)

        with open(os.path.join(config.out_dir, "config.txt"), "w") as text_file:
            text_file.write("%s" % config)

        if config.test_code:
            exit(0)


def main():
    config = parse_config()
    # config, net, optimiser = setup(config)
    net = None  # TODO
    optimiser = None  # TODO
    if config.evaluate_clusters:
        evaluate_clusters(config, net)
    else:
        train(config, net, optimiser)


if __name__ == '__main__':
    main()
