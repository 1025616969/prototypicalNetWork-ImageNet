import argparse
from main import *
parser=argparse.ArgumentParser(description='原型网络走一走')

# data args


parser.add_argument('--data.way', type=int, default=20, metavar='WAY',
                    help="number of classes per episode (default: 60)")
parser.add_argument('--data.shot', type=int, default=5, metavar='SHOT',
                    help="number of support examples per class (default: 5)")
parser.add_argument('--data.query', type=int, default=5, metavar='QUERY',
                    help="number of query examples per class (default: 5)")
parser.add_argument('--data.test_way', type=int, default=5, metavar='TESTWAY',
                    help="number of classes per episode in test. 0 means same as data.way (default: 5)")
parser.add_argument('--data.test_shot', type=int, default=1, metavar='TESTSHOT',
                    help="number of support examples per class in test. 0 means same as data.shot (default: 0)")
parser.add_argument('--data.test_query', type=int, default=15, metavar='TESTQUERY',
                    help="number of query examples per class in test. 0 means same as data.query (default: 15)")
parser.add_argument('--data.train_episodes', type=int, default=5, metavar='NTRAIN',
                    help="number of train episodes per epoch (default: 100)")
parser.add_argument('--data.test_episodes', type=int, default=3, metavar='NTEST',
                    help="number of test episodes per epoch (default: 100)")

parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")



# train args
parser.add_argument('--train.epoches', type=int, default=3, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
default_weight_decay = 0.0
parser.add_argument('--train.weight_decay', type=float, default=default_weight_decay, metavar='WD',
                    help="weight decay (default: {:f})".format(default_weight_decay))


# log args
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))

#mode
parser.add_argument('--run_mode',type=str,default='train',help="运行方式，输入'test'则加载模型进行测试，默认为'train'。")

args = vars(parser.parse_args())

main(args)