import os
import sys
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import SGD, Adam
import time
import argparse

import torch
import numpy as np
import random
import os
from torch.backends import cudnn
from torchvision.datasets import CIFAR100, ImageFolder, CIFAR10, SVHN,FashionMNIST,MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import get_dataloaders

from networks.resnet import resnet18, resnet34
# sys.path.append("../")
from networks.wresnet import wrn_16_4
from networks.lenet import LeNet5
from networks.conv3 import Conv3
from knockoff import knockoff_
from DefencePlugin import PatchPlugin


torch.cuda.set_device(1)		#设置1号卡，识别当前空间的所有卡

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def my_test(model, data_loader):
    """
    test accuracy
    """
    model = model.cuda()
    model.eval()
    correct = 0.0
    count = 0
    with torch.no_grad():
        for (x, y) in data_loader:
            x, y = x.cuda(),y.cuda()
            pred = model(x)
            pred_class = torch.argmax(pred, dim=1)
            correct += (pred_class == y).sum().item()
            count += y.shape[0]
        acc = correct / count #len(data_loader.dataset)
    return acc

def enable_gpu_benchmarking():
    if torch.cuda.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

def get_loader(dataset,bs,data_dir,train=True):
    if dataset=='cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        cifar10_test_set = CIFAR10(data_dir, download=True,
                                   train=train,
                                   transform=transform)
        loader = DataLoader(cifar10_test_set, batch_size=bs, num_workers=8)

    if dataset=='cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
        ])
        cifar100_test_set = CIFAR100(data_dir, download=True,
                                   train=train,
                                   transform=transform)
        loader = DataLoader(cifar100_test_set, batch_size=bs, num_workers=8)

    if dataset == 'mnist':
        data_test = MNIST(data_dir,
                          train=train,
                          transform=transforms.Compose([
                              transforms.Resize((32, 32)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))
                          ]), download=True)
        loader = DataLoader(data_test, batch_size=bs,num_workers=8)

    if dataset=='fashion':
        data_test = FashionMNIST(data_dir,
                                 train=train,
                                 transform=transforms.Compose([
                                     transforms.Resize((32, 32)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.1307,), (0.3081,))
                                 ]), download=True)
        loader = DataLoader(data_test, batch_size=bs, num_workers=8)

    if dataset=='flowers':
        loader = get_dataloaders('../datasets/flowers/validation/',bs)

    return loader

def get_surrogate_dataloader(dataset,bs,data_dir):
    if dataset == 'mnist':
        data_train = MNIST(data_dir,
                           train=True,
                           transform=transforms.Compose([
                               transforms.Resize((32, 32)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]), download=True)
        data_train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=8)

    if dataset == 'fashion':
        data_train = FashionMNIST(data_dir,
                                  train=True,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]), download=True)
        data_train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=8)

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10(data_dir,train=True,transform=transform_train)
        data_train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=8)

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        data_train = CIFAR100(data_dir,train=True, transform=transform_train)
        data_train_loader = DataLoader(data_train, batch_size=bs, shuffle=True, num_workers=8)

    if dataset=='indoor':
        data_train_loader = get_dataloaders('../datasets/Images/', bs)

    return data_train_loader


def get_statistical(victim_dir):
    # =============Loading dataset and statistical value =================
    if 'cifar10' in victim_dir and 'cifar100' not in victim_dir:
        mean = np.load('../statistical_value/wrn16_4_cifar10_mean.npy')
        cova = np.load('../statistical_value/wrn16_4_cifar10_cova.npy')
    if 'cifar100' in victim_dir:
        mean = np.load('../statistical_value/wrn16_4_cifar100_mean.npy')
        cova = np.load('../statistical_value/wrn16_4_cifar100_cova.npy')
    if 'mnist' in victim_dir:
        mean = np.load('../statistical_value/lenet_mnist_mean.npy')
        cova = np.load('../statistical_value/lenet_mnist_cova.npy')
    if 'fashion' in victim_dir:
        mean = np.load('../statistical_value/lenet_fashionMnist_mean.npy')
        cova = np.load('../statistical_value/lenet_fashionMnist_cova.npy')
    if 'flowers17' in victim_dir:
        mean = np.load('../statistical_value/resnet18_flowers17_mean.npy')
        cova = np.load('../statistical_value/resnet18_flowers17_cova.npy')
    return mean,cova

set_seed(2023)
enable_gpu_benchmarking()
model_choices = ['conv3','lenet','wres16','resnet18']
ds_choices = ['mnist','fashion','cifar10','cifar100','flowers17','indoor67']

def main():
    parser = argparse.ArgumentParser(description="attack ensemble")

    parser.add_argument(
        "--defence", type=bool, default=True, help="Start DFS defense"
    )

    # parser.add_argument(
    #     "--defence" , action="store_true", help="Start DFS defense"
    # )
    parser.add_argument('--inter_thre', type=float, default=5,  #fashion-150 ; mnist-1500  cifar10-5   cifar100-50
                        help='The manufacturer is generally set on the server side')
    parser.add_argument('--dims', type=int, default=512,
                        choices=[1024, 256, 512, 120, 2048],  # mnist 120  cifar10 256  cifar100 256  fashionmnist 120    conv3-512
                        help=('Dimensionality of Inception features to use. '))

    parser.add_argument('--window_size', type=int, default=0) # window_size = 64 && query_batch_size=1
    parser.add_argument("--query_batch_size", type=int, default=64, help="batch size") # window_size = 0 && query_batch_size=64
    parser.add_argument("--clone_batch_size", type=int, default=90, help="batch size")

    parser.add_argument(
        "--pred_type",
        type=str,
        default="hard",
        help="specify if the target model outputs hard/soft label predictions",
        choices=["hard", "soft"],
    )

    parser.add_argument('--victim_dir',type=str,default='../pretrained/resnet18_flowers17.pth')
    parser.add_argument(
        "--dataset_tar",
        type=str,
        default="flowers17",
        help="target dataset",
        choices=ds_choices,
    )
    parser.add_argument(
        "--dataset_sur",
        type=str,
        default="indoor67",
        help="surrogate dataset used to query the target",
        choices=ds_choices,
    )

    parser.add_argument(
        "--model_tar",
        type=str,
        default="resnet18",   #wres16  lenet  conv3
        choices=model_choices,
        help="Target model type",
    )


    parser.add_argument(
        "--model_sur",
        type=str,
        default="resnet18",
        choices=model_choices,
        help="Surrogate/Clone model type",
    )
    parser.add_argument(
        "--opt", type=str, default="sgd", choices=["sgd", "adam"], help="Optimizer"
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")

    # parser.add_argument(
    #     "--n_models", type=int, default=1, help="number of models to train"
    # )
    parser.add_argument(
        "--n_seed", type=int, default=2023, help="number of seed examples"
    )
    parser.add_argument(
        "--data_dir", type=str, default='../datasets', help="path of datasets"
    )

    parser.add_argument("--budget", type=int, default=50000, help="Query budget")
    parser.add_argument(
        "--aug_rounds", type=int, default=6, help="number of augmentation rounds"
    )
    # parser.add_argument(
    #     "--id_hash", type=str, default="32", help="name of the hash experiment"
    # )
    parser.add_argument(
        "--quantize", action="store_true", help="Enable input quantization"
    )
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument(
        "--pretrained", action="store_true", help="use pretrained model"
    )
    parser.add_argument("--disable_pbar", action="store_true", help="Disable pbar")
    parser.add_argument("--augment", action="store_true", help="Use Pretrained model")
    parser.add_argument(
        "--attack",
        type=str,
        default="knockoff",
        help="attack",
        choices=["knockoff"],
    )
    parser.add_argument(
        "--n_adaptive_queries", type=int, default=5, help="number of adaptive queries"
    )
    parser.add_argument(
        "--exp_id", type=str, default="default", help="name of the experiment"
    )
    parser.add_argument(
        "--adaptive_mode",
        type=str,
        default="none",
        help="Adaptive attack mode",
        choices=["none", "normal", "ideal_attack", "ideal_defense", "normal_sim"],
    )

    args = parser.parse_args()
    print(args)
    path_exp = f"./exp/{args.dataset_tar}/{args.exp_id}/"

    #========加载训练集和测试集========

    if 'flowers17' not in args.victim_dir:
        dataloader_test = get_loader(args.dataset_tar,args.query_batch_size,args.data_dir,False)
        dataloader_train = get_loader(args.dataset_tar,64, args.data_dir, True) #args.window_size
        dataloader_sur = get_surrogate_dataloader(args.dataset_sur, args.query_batch_size, args.data_dir)
    else:
        _ ,dataloader_test = get_dataloaders('flowers17',batch_size=args.query_batch_size)
        dataloader_train,_ = get_dataloaders('flowers17',batch_size=64) #args.window_size
        dataloader_sur,_ = get_dataloaders('indoor67',batch_size=args.query_batch_size)

    dataloader,_ = next(iter(dataloader_train))

    if 'wrn16_4' in args.victim_dir and 'cifar10' in args.victim_dir:
        victim = wrn_16_4(num_classes=10).cuda()
        attacker = wrn_16_4(num_classes=10).cuda()

    if 'wrn16_4' in args.victim_dir and 'cifar100' in args.victim_dir:
        victim = wrn_16_4(num_classes=100).cuda()
        attacker = wrn_16_4(num_classes=100).cuda()

    if 'lenet' in args.victim_dir:
        victim = LeNet5().cuda()
        attacker = LeNet5().cuda()

    if 'conv3' in args.victim_dir:
        victim = Conv3(32,1,10).cuda()
        attacker = Conv3(32,1,10).cuda()

    if 'resnet18' in args.victim_dir:
        victim = resnet18(num_classes=17).cuda()
        attacker = resnet18(num_classes=17).cuda()



    victim.load_state_dict(torch.load(args.victim_dir, map_location='cpu')['net'])
    victim.eval()

    T = victim
    S = attacker

    if args.defence:
        mean,cova = get_statistical(args.victim_dir)
        def_victim = PatchPlugin(victim, args.query_batch_size, 0 , args.dims, mean, cova, args.num_workers,
                                 args.inter_thre,args.window_size,dataloader)
        T = def_victim



    acc_tar = my_test(T, dataloader_test)
    print("* Loaded Target Model *")
    print("Target Accuracy: {:.2f}%\n".format(100 * acc_tar))

    if args.opt == "sgd":
        opt = SGD(S.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        sch = CosineAnnealingLR(opt, args.epochs, last_epoch=-1)
    elif args.opt == "adam":
        opt = Adam(S.parameters(), lr=args.lr)
        sch = None

    if args.attack == "knockoff":
        knockoff_(
            T,
            S,
            dataloader_sur,
            dataloader_test,
            opt,
            # sch,
            acc_tar,
            args.clone_batch_size,
            args.epochs,
            args.disable_pbar,
            args.budget,
            pred_type=args.pred_type,
            adaptive_mode=args.adaptive_mode,
            n_adaptive_queries=args.n_adaptive_queries,
        )

    else:
        sys.exit("Unknown Attack {}".format(args.attack))

    savedir_clone = path_exp + "clone/"
    if not os.path.exists(savedir_clone):
        os.makedirs(savedir_clone)

    torch.save(S.state_dict(), savedir_clone + "{}.pt".format(args.attack))
    print("* Saved Sur model * ")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    runtime = end - start
    print("Runtime: {:.2f} s".format(runtime))
