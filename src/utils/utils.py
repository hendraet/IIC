import argparse


def get_std_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_ind", type=int)

    parser.add_argument("--arch", type=str, required=True, help="selects architecture of the models")
    parser.add_argument("--restart", dest="restart", default=False, action="store_true")
    parser.add_argument("--restart_from_best", dest="restart_from_best", default=False, action="store_true")
    parser.add_argument("--test_code", dest="test_code", default=False, action="store_true")
    parser.add_argument("--out_root", type=str)

    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_schedule", type=int, nargs="+", default=[])
    parser.add_argument("--lr_mult", type=float, default=0.1)

    return parser
