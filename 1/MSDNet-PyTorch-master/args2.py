import os
import glob
import time
import argparse


model_names = ['msdnet']
print('5')


def path_check(string): #checks for valid path
    if os.path.exists(string):
        return string
    else:
        raise FileNotFoundError(string)
print('3')

def msd_args():
	'''
	parser = argparse.ArgumentParser(description="script for running pytorch-onnx tests")
	parser.add_argument('--model',choices=['brn','lenet'],
                        help='choose the model')
	parser.add_argument('--trained_path', type=path_check,
                        help='path to trained model')
	parser.add_argument('--save_name', type=str,
                        help='path to trained model')
	args1 = parser.parse_args()
	print(args1)
	return args1
	'''

	arg_parser = argparse.ArgumentParser(
		        description='Image classification PK main script')

#	exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
	arg_parser.add_argument('--save_name', type=str, default='save/default-{}'.format(time.time()),metavar='SAVE',
		               help='path to the experiment logging directory'
		               '(default: save/debug)')
	arg_parser.add_argument('--resume', action='store_true',
		               help='path to latest checkpoint (default: none)')
	arg_parser.add_argument('--evalmode', default=None,
		               choices=['anytime', 'dynamic'],
		               help='which mode to evaluate')
	arg_parser.add_argument('--trained_path', type=path_check, metavar='PATH', default=None,
		               help='path to saved checkpoint (default: none)')
	arg_parser.add_argument('--print-freq', '-p', default=10, type=int,
		               metavar='N', help='print frequency (default: 100)')
	arg_parser.add_argument('--model', choices=['brn','lenet'],
		               help='choose model')
	arg_parser.add_argument('--seed', default=0, type=int,
		               help='random seed')
	arg_parser.add_argument('--gpu', default=None, type=str, help='GPU available.')
	
	# dataset related
#	data_group = arg_parser.add_argument_group('data', 'dataset setting')
	arg_parser.add_argument('--data', metavar='D', default='cifar10',
		                choices=['cifar10', 'cifar100', 'ImageNet'],
		                help='data to work on')
	arg_parser.add_argument('--data-root', metavar='DIR', default='data',
		                help='path to dataset (default: data)')
	arg_parser.add_argument('--use-valid', action='store_true',
		                help='use validation set or not')
	arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
		                help='number of data loading workers (default: 4)')

	# model arch related
#	arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
	arg_parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
		                type=str, choices=model_names,
		                help='model architecture: ' +
		                ' | '.join(model_names) +
		                ' (default: msdnet)')
	arg_parser.add_argument('--reduction', default=0.5, type=float,
		                metavar='C', help='compression ratio of DenseNet'
		                ' (1 means dot\'t use compression) (default: 0.5)')

	# msdnet config
	arg_parser.add_argument('--nBlocks', type=int, default=1)
	arg_parser.add_argument('--nChannels', type=int, default=32)
	arg_parser.add_argument('--base', type=int,default=4)
	arg_parser.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
	arg_parser.add_argument('--step', type=int, default=1)
	arg_parser.add_argument('--growthRate', type=int, default=6)
	arg_parser.add_argument('--grFactor', default='1-2-4', type=str)
	arg_parser.add_argument('--prune', default='max', choices=['min', 'max'])
	arg_parser.add_argument('--bnFactor', default='1-2-4')
	arg_parser.add_argument('--bottleneck', default=True, type=bool)


	# training related
#	optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')

	arg_parser.add_argument('--epochs', default=300, type=int, metavar='N',
		                 help='number of total epochs to run (default: 164)')
	arg_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
		                 help='manual epoch number (useful on restarts)')
	arg_parser.add_argument('-b', '--batch-size', default=64, type=int,
		                 metavar='N', help='mini-batch size (default: 64)')
	arg_parser.add_argument('--optimizer', default='sgd',
		                 choices=['sgd', 'rmsprop', 'adam'], metavar='N',
		                 help='optimizer (default=sgd)')
	arg_parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
		                 metavar='LR',
		                 help='initial learning rate (default: 0.1)')
	arg_parser.add_argument('--lr-type', default='multistep', type=str, metavar='T',
		                help='learning rate strategy (default: multistep)',
		                choices=['cosine', 'multistep'])
	arg_parser.add_argument('--decay-rate', default=0.1, type=float, metavar='N',
		                 help='decay rate of learning rate (default: 0.1)')
	arg_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
		                 help='momentum (default=0.9)')
	arg_parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
		                 metavar='W', help='weight decay (default: 1e-4)')

	print('6')
	args1 = arg_parser.parse_args()


	if args1.gpu:
	    os.environ["CUDA_VISIBLE_DEVICES"] = args1.gpu

	args1.grFactor = list(map(int, args1.grFactor.split('-')))
	args1.bnFactor = list(map(int, args1.bnFactor.split('-')))
	args1.nScales = len(args1.grFactor)

	if args1.use_valid:
	    args1.splits = ['train', 'val', 'test']
	else:
	    args1.splits = ['train', 'val']

	if args1.data == 'cifar10':
	    args1.num_classes = 10
	elif args1.data == 'cifar100':
	    args1.num_classes = 100
	else:
	    args1.num_classes = 1000

	return args1



		
'''
if main.model == 'brn':
    brn_main(md_pth=args.trained_path, save_name=args.save_name)
elif main.model == 'lenet':
    print("ignorning model path provided")
    lenet_main()
'''

