import os
import logging
import argparse

parser = argparse.ArgumentParser()
#tarnet, cfrnet, dragonnet, dragonnetTR
#IHDP, SIPP
parser.add_argument("--model", type=str, nargs="?", default = "dragonnetTR")
parser.add_argument("--dataset", type=str, default = 'IHDP')

parser.add_argument("--num_units_rep", type=int, default=200)
parser.add_argument("--num_units_hypo", type=int, default=100)
parser.add_argument("--actv", type=str, default='elu')
parser.add_argument("--kernel_init", type=str, default='RandomNormal')
parser.add_argument("--kernel_reg", type=str, default='L2')
parser.add_argument("--reg_param", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--nesterov", type=bool, default=True)
parser.add_argument("--val_split", type=float, default=0.2)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epoch", type=int, default=300)
parser.add_argument("--verbose", type=int, default=1)

# parser.add_argument("--save_path", type=str, default="save/test/")
# parser.add_argument("--save_path_dataset", type=str, default="save/")
#parser.add_argument("--cuda", action="store_true")

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, opts.__dict__[key]).center(80))
    print("=" * 80)

arg = parser.parse_args()
#print(arg)
print_opts(arg)

model = arg.model
dataset = arg.dataset
num_units_rep = arg.num_units_rep
num_units_hypo = arg.num_units_hypo
actv = arg.actv
kernel_init = arg.kernel_init
kernel_reg = arg.kernel_reg
reg_param = arg.reg_param
verbose = arg.verbose
lr = arg.lr
momentum = arg.momentum
nesterov = arg.nesterov
val_split = arg.val_split
batch_size = arg.batch_size
num_epoch = arg.num_epoch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')#,filename='save/logs/{}.log'.format(str(name)))
collect_stats = False
