from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import os
import sys
import time
import logging
import torch.backends.cudnn as cudnn
from attacks.networks.resnet import resnet18, resnet34
from attacks.networks.wresnet import wrn_16_4
from attacks.networks.lenet import LeNet5
from torchvision.datasets import CIFAR100, ImageFolder, CIFAR10, SVHN,FashionMNIST,MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np
import warnings
import torch
import argparse
from attacks.DefencePlugin import PatchPlugin
from attacks.dataset import get_dataloaders
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='KD')
# various path
parser.add_argument('--save_root', type=str, default='./result/', help='models and logs are saved here')
parser.add_argument('--data_dir',type=str,default='./datasets/')

parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--T',type=int,default=3,help='')


parser.add_argument('--test',type=bool,default=True)
parser.add_argument('--device', type=str, default=0,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--victim_dir', type=str, default='./pretrained/resnet18_flowers17.pth')
parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
parser.add_argument('--dims', type=int, default=512,
                    choices=[1024,256,512,120,2048],   #mnist 120  cifar10 256  cifar100 256  fashionmnist
                    help=('Dimensionality of Inception features to use. '
                                      'By default, uses pool3 features'))
parser.add_argument('--inter_thre', type=float, default=5, help='The manufacturer is generally set on the server side')
parser.add_argument('--tau', type=float, default=0.0005, help='Percentage of accuracy reduction allowed')

# cifar10
# others
parser.add_argument('--seed', type=int, default=2023, help='random seed')
parser.add_argument('--note', type=str, default='default', help='note for this run')
args, unparsed = parser.parse_known_args()
args.save_root = os.path.join(args.save_root, args.note)

if not os.path.exists(args.save_root):
	os.makedirs(args.save_root)

def get_dataloader(dataset,train=False):
	if dataset=='cifar10':
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
		])
		cifar10_test_set = CIFAR10(args.data_dir, download=True,
								   train=train,
								   transform=transform)

		loader = DataLoader(cifar10_test_set, batch_size=args.batch_size, num_workers=args.num_workers,
										 shuffle=True)

	if dataset=='cifar100':
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
		])
		cifar100_test_set = CIFAR100(args.data_dir, download=True,
								   train=train,
								   transform=transform)

		loader = DataLoader(cifar100_test_set, batch_size=args.batch_size, num_workers=args.num_workers,
										 shuffle=True)

	if dataset == 'mnist':
		data_test = MNIST(args.data_dir,
						  train=train,
						  transform=transforms.Compose([
							  transforms.Resize((32, 32)),
							  transforms.ToTensor(),
							  transforms.Normalize((0.1307,), (0.3081,))
						  ]), download=True)

		loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=8)

	if dataset=='fashionMnist':
		data_test = FashionMNIST(args.data_dir,
								 train=train,
								 transform=transforms.Compose([
									 transforms.Resize((32, 32)),
									 transforms.ToTensor(),
									 transforms.Normalize((0.1307,), (0.3081,))
								 ]), download=True)

		loader = DataLoader(data_test, batch_size=args.batch_size, num_workers=8)

	if dataset == 'flowers17':
		loader, _ = get_dataloaders('flowers17', batch_size=args.batch_size)
	return loader

def get_surrogate_dataloader(dataset):
	if dataset == 'mnist':
		data_train = MNIST(args.data_dir,
						   transform=transforms.Compose([
							   transforms.Resize((32, 32)),
							   transforms.ToTensor(),
							   transforms.Normalize((0.1307,), (0.3081,))
						   ]), download=True)

		data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

	if dataset == 'fashionMnist':
		data_train = FashionMNIST(args.data_dir,
								  transform=transforms.Compose([
									  transforms.Resize((32, 32)),
									  transforms.ToTensor(),
									  transforms.Normalize((0.1307,), (0.3081,))
								  ]), download=True)

		data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

	if dataset == 'cifar10':
		transform_train = transforms.Compose([
			# transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		data_train = CIFAR10(args.data_dir,transform=transform_train)
		data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=8)

	if dataset == 'cifar100':
		transform_train = transforms.Compose([
			# transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
		])
		data_train = CIFAR100(args.data_dir, transform=transform_train)
		data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

	if dataset == 'indoor67':
		data_train_loader, _ = get_dataloaders('indoor67', batch_size=args.batch_size)

	return data_train_loader

def model_test(test_loader,student,criterion):
	student.eval()
	total_correct = 0
	avg_loss = 0.0
	count = 0
	with torch.no_grad():
		for i, (images, labels) in enumerate(test_loader):
			if(device=='cuda'):
				images, labels = images.cuda(), labels.cuda()
			with torch.no_grad():
				output = student(images)
				avg_loss += criterion(output, labels).sum()
			pred = output.data.max(1)[1]
			count+=labels.size(0)
			total_correct += pred.eq(labels.data.view_as(pred)).sum()

	avg_loss /= count
	acc = float(total_correct) / count
	# print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))
	return avg_loss.data.item() ,acc

def main():
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	crit_CE = torch.nn.CrossEntropyLoss().cuda()
	if device=='cuda':
		torch.cuda.manual_seed(args.seed)
		cudnn.enabled = True
		cudnn.benchmark = True
	logging.info("args = %s", args)

	# ----------- Network Initialization --------------
	if(device=='cuda'):
		if 'wrn16_4' in args.victim_dir and 'cifar10' in args.victim_dir:
			victim = wrn_16_4(num_classes=10).cuda()
		if 'wrn16_4' in args.victim_dir and 'cifar100' in args.victim_dir:
			victim = wrn_16_4(num_classes=100).cuda()
		if 'lenet' in args.victim_dir:
			victim = LeNet5().cuda()
		if 'flowers17' in args.victim_dir:
			victim = resnet18(num_classes=17).cuda()
	else:
		assert "Please Use GPU ！"

	victim.load_state_dict(torch.load(args.victim_dir,map_location='cpu')['net'])
	victim.eval()

	#=============Loading dataset and statistical value =================
	if 'cifar10' in args.victim_dir and 'cifar100' not in args.victim_dir:
		# test_loader = get_dataloader('cifar10',train=False)
		train_loader = get_dataloader('cifar10', train=True)
		mean = np.load('./statistical_value/wrn16_4_cifar10_mean.npy')
		cova = np.load('./statistical_value/wrn16_4_cifar10_cova.npy')
	if 'cifar100' in args.victim_dir:
		# test_loader = get_dataloader('cifar100',train=False)
		train_loader = get_dataloader('cifar100', train=True)
		mean = np.load('./statistical_value/wrn16_4_cifar100_mean.npy')
		cova = np.load('./statistical_value/wrn16_4_cifar100_cova.npy')
	if 'mnist' in args.victim_dir:
		# test_loader = get_dataloader('mnist',train=False)
		train_loader = get_dataloader('mnist', train=True)
		mean = np.load('./statistical_value/lenet_mnist_mean.npy')
		cova = np.load('./statistical_value/lenet_mnist_cova.npy')
	if 'fashion' in args.victim_dir:
		# test_loader = get_dataloader('fashionMnist',train=False)
		train_loader = get_dataloader('fashionMnist', train=True)
		mean = np.load('./statistical_value/lenet_fashionMnist_mean.npy')
		cova = np.load('./statistical_value/lenet_fashionMnist_cova.npy')
	if 'flowers17' in args.victim_dir:
		mean = np.load('./statistical_value/resnet18_flowers17_mean.npy')
		cova = np.load('./statistical_value/resnet18_flowers17_cova.npy')
		train_loader = get_dataloader('flowers17', train=True)


	_ , org_acc = model_test(train_loader,victim,crit_CE)
	print("train dataset org_acc:",org_acc)
	# _, acc = model_test(test_loader, victim, crit_CE)
	# print("Test datasset acc: ",acc)

	for thr in np.arange(8,15,0.1):
		args.inter_thre = thr
		def_victim = PatchPlugin(victim, args.batch_size, args.device , args.dims, mean, cova, args.num_workers,args.inter_thre)
		lo,acc = model_test(train_loader,def_victim,crit_CE)
		# print('--------  '+str(round(acc,4))+'  --------')
		print(str(thr)+" "+str(round(acc, 4)))
		# if(acc>=(org_acc-args.tau)):
		# 	print("==================================")
		# 	print(args.inter_thre,acc)
		# 	# _, acc = model_test(test_loader, def_victim, crit_CE)
		# 	# print(acc)
		# 	print("==================================")

if __name__ == '__main__':
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	main()
