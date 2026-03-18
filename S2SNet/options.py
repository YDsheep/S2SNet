import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=448, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

parser.add_argument('--rgb_root', type=str, default='datasets/ORSSD/train/RGB/',
                    help='the training rgb images root')
parser.add_argument('--gt_root', type=str, default='datasets/ORSSD/train/GT/',
                    help='the training GT images root')
parser.add_argument('--test_rgb_root', type=str, default='datasets/ORSSD/test/RGB/',
                    help='the test rgb images root')
parser.add_argument('--test_gt_root', type=str, default='datasets/ORSSD/test/GT/',
                    help='the test GT images root')

# parser.add_argument('--rgb_root', type=str, default='datasets/4199/train/RGB/',
#                     help='the training rgb images root')
# parser.add_argument('--gt_root', type=str, default='datasets/4199/train/GT/',
#                     help='the training GT images root')
# parser.add_argument('--test_rgb_root', type=str, default='datasets/4199/test/RGB/',
#                     help='the test rgb images root')
# parser.add_argument('--test_gt_root', type=str, default='datasets/4199/test/GT/',
#                     help='the test GT images root')


parser.add_argument('--save_path', type=str, default='save/ORSSD/', help='the path to save models and logs')

parser.add_argument('--pre', type=str, default='models/VMamba/vssmtiny_dp01_ckpt_epoch_292.pth', help='pre')
parser.add_argument('--cfg', type=str, default='models/VMamba/vmambav2v_tiny.yaml', help='cfg')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
opt = parser.parse_args()
