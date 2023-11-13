"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pathlib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torch.nn as nn
import numpy as np
import torch
import torchvision.transforms as TF
from PIL import Image
from scipy import linalg
from torch.nn import functional
import math
from torchvision import transforms
from networks.lenet import LeNet5
from networks.wresnet import wrn_16_4
from networks.resnet import resnet18
from torchvision.datasets import CIFAR10,CIFAR100,MNIST,FashionMNIST
from sklearn.neighbors import KernelDensity
from collections import deque

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# from inception import InceptionV3



parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=256,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,default=0,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=0,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=120,
                    choices=[1024,256,512,120,2048],   #mnist 120  cifar10 256  cifar100 256  fashionmnist
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--model_path', type=str,default='../result/fashionMnist/lenet_fashionMnist.pth',
                    help=('Paths to the generated images or '  #../result/wrn16_4_cifar100/wrn16_4_cifar100.pth  mnist/lenet_mnist.pth
                          'to .npz statistic files'))
parser.add_argument('--num_class', type=int, default=10)
parser.add_argument('--dataset', type=str, default='../datasets/')
args, unparsed = parser.parse_known_args()

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset, transforms=None):
        self.dataset = img_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        img = self.dataset[i]
        # img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img



class KDEPlugin(nn.Module):
    def __init__(self,train_loader,victim,inter_thre):
        super(KDEPlugin,self).__init__()

        self.victim = victim
        self.inter_thre = inter_thre
        feature = []
        for i, (img, x) in enumerate(train_loader):
            img = img.cuda()
            feat = victim(img.detach(), return_rep=True)[1]  # (batch_size,512)
            feature.append(feat.detach())
        feature = torch.cat(feature, dim=0)
        feature = feature.detach().cpu().numpy()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.26).fit(feature)
        self.kde = kde


    def forward(self,imgs_tensor):
        imgs_tensor = imgs_tensor.cuda()
        output,feat = self.victim(imgs_tensor.detach(), return_rep=True)  # (batch_size,512)
        feat = feat.detach().cpu().numpy()
        log_density = self.kde.score_samples(feat)

        # print(log_density)

        is_ood = log_density<self.inter_thre
        output[is_ood] = torch.randn(len(output[0]),dtype=float,device='cuda').float().cuda()
        return output


class FMD():

    def __init__(self,model,batch_size,device,dims,mean,cova,num_workers):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.mean = mean
        self.cova = cova
        self.num_workers = num_workers


    def get_activations(self,images_tensor, model, batch_size=50, dims=2048, device='cpu',
                        num_workers=1):
        """Calculates the activations of the pool_3 layer for all images.

        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                         Make sure that the number of samples is a multiple of
                         the batch size, otherwise some samples are ignored. This
                         behavior is retained to match the original FID score
                         implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers

        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
           activations of the given tensor when feeding inception with the
           query tensor.
        """
        model.eval()

        # batch_size = len(images_tensor)
        # dataset = ImageDataset(images_tensor, transforms=TF.ToTensor()) # 拿到dataset转换成tensor就行
        # dataloader = torch.utils.data.DataLoader(dataset,
        #                                          batch_size=batch_size,
        #                                          shuffle=False,
        #                                          drop_last=False,
        #                                          num_workers=num_workers)

        pred_arr = np.empty((len(images_tensor), dims))

        start_idx = 0

        # for batch in dataloader:
        batch = images_tensor.cuda()

        with torch.no_grad():
            pred = model(batch,return_rep=True)[1]

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred.cpu()

        return pred_arr


    def calculate_frechet_distance(self,mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            # print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)


    def calculate_activation_statistics(self,images_data, model, batch_size=50, dims=2048,
                                        device='cpu', num_workers=1):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                         batch size batch_size. A reasonable batch size
                         depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        -- num_workers : Number of parallel dataloader workers

        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                   the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                   the inception model.
        """
        act = self.get_activations(images_data, model, batch_size, dims, device, num_workers)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma , act

    def compute_statistics_of_path(self,images_data, model, batch_size, dims, device,
                                   num_workers=1):
        m, s, a = self.calculate_activation_statistics(images_data, model, batch_size,
                                               dims, device, num_workers)

        return m, s, a


    def calculate_fid(self,images_data):

        m1 = self.mean
        s1 = self.cova

        m2, s2 ,feat = self.compute_statistics_of_path(images_data, self.model, self.batch_size,
                                            self.dims, self.device, self.num_workers)

        fid_value = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value,feat

    def calculate_single_cosin(self,images_tensor,prototype):
        # prototype.shape =(10*512)
        feat = self.model(images_tensor, return_rep=True)[1] # feat.shape=(batch_size,512)
        dist = functional.cosine_similarity(images_tensor,feat.T) #(256,10)
        dist_min = torch.min(dist,dim=1)
        return dist_min



class PatchPlugin(nn.Module):

    # 输入tensor，不需要转为Image
    def __init__(self,victime,batch_size,device,dims,mean,cova,num_workers,inter_thre,window_size=0,trainloader=None):
        super(PatchPlugin, self).__init__()
        self.teacher = victime
        self.fmd = FMD(victime,batch_size,device,dims,mean,cova,num_workers)
        self.inter_thre = inter_thre
        self.window_size = window_size #xs = torch.cat((xs, x.cpu()), dim=0)
        self.window = deque(maxlen=window_size)

        if(window_size!=0):
            # output , feature = self.teacher(trainloader.cuda(), True)
            # for f in feature:
            #     self.window.append(f)
            for tensor in trainloader:
                self.window.append(tensor.cuda().reshape(1,tensor.shape[0],tensor.shape[1],tensor.shape[1]))

    def input(self,imgs_tensor,signal=False):

        if(self.window_size==0):
            # =========batch=========
            defenced = 0
            tensor = imgs_tensor.clone()
            # images = self.fmd.transform_invert(tensor)
            fmd_val,feat = self.fmd.calculate_fid( tensor )
            print(fmd_val)
            if fmd_val>self.inter_thre:
                # feat = feat + torch.randn(feat.shape,dtype=float,device='cuda').cuda()
                feat = torch.randn(feat.shape,dtype=float,device='cuda').cuda()
                # feat = torch.zeros(feat.shape, dtype=float, device='cuda').cuda()
                defenced = 1
            if signal:
                return self.teacher.classifier(torch.tensor(feat,device='cuda').float()), defenced
            else:
                return self.teacher.classifier(torch.tensor(feat, device='cuda').float())

        else:
            # =========single=========
            # print((1,imgs_tensor.shape[1],imgs_tensor.shape[2],imgs_tensor.shape[2]))
            # print(imgs_tensor.shape)
            img_tensor = imgs_tensor.reshape(1,imgs_tensor.shape[1],imgs_tensor.shape[2],imgs_tensor.shape[2]).cuda()
            # print(img_tensor.shape)
            self.window.popleft()
            self.window.append(img_tensor)
            imgs_tensor = list(self.window)

            defenced = 0
            # tensor = imgs_tensor.clone()
            # images = self.fmd.transform_invert(tensor)
            tensor = torch.cat(imgs_tensor,dim=0).cuda()
            fmd_val, feat = self.fmd.calculate_fid(tensor)

            feat = feat[self.window_size-1]
            feat = (feat).reshape(1,-1)

            # print(fmd_val)

            if fmd_val > self.inter_thre:
                # print(fmd_val)
                # feat = feat + torch.randn(feat.shape,dtype=float,device='cuda').cuda()
                feat = torch.randn(feat.shape, dtype=float, device='cuda').cuda()
                # feat = torch.zeros(feat.shape, dtype=float, device='cuda').cuda()
                defenced = 1

            if signal:
                return self.teacher.classifier(torch.tensor(feat, device='cuda').float()), defenced
            else:
                return self.teacher.classifier(torch.tensor(feat, device='cuda').float())


    def forward(self, x,signal=False):
        return self.input(x,signal)


def main():

    def load_undef_victim():
        if 'cifar10' in args.model_path or 'cifar100' in args.model_path:
            victim = wrn_16_4(args.num_class).cuda()
        if 'lenet' in args.model_path:
            victim = LeNet5().cuda()

        victim.load_state_dict(torch.load(args.model_path, map_location='cpu')['net'])
        victim.eval()

        return victim

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.2, 0.2, 0.2))])

    transform_one = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    undef_victim = load_undef_victim()

    #----------loading dataset and statistical value----------
    if 'cifar10' in args.model_path:
        dataset = CIFAR10(root=args.dataset, train=False,
                          download=True, transform=transform)
        mean = np.load('./results/statistical_value/wrn16_4_cifar10_mean.npy')
        cova = np.load('./results/statistical_value/wrn16_4_cifar10_cova.npy')
    if 'cifar100' in args.model_path:
        dataset = CIFAR100(root=args.dataset, train=False,
                           download=True, transform=transform)
        mean = np.load('./results/statistical_value/wrn16_4_cifar100_mean.npy')
        cova = np.load('./results/statistical_value/wrn16_4_cifar100_cova.npy')
    if 'mnist' in args.model_path:
        dataset = MNIST(root=args.dataset, train=False,
                        download=True, transform=transform_one)
        mean = np.load('./results/statistical_value/lenet_mnist_mean.npy')
        cova = np.load('./results/statistical_value/lenet_mnist_cova.npy')
    if 'fashionMnist' in args.model_path:
        dataset = FashionMNIST(root=args.dataset, train=False,
                               download=True, transform=transform_one)
        mean = np.load('./results/statistical_value/lenet_fashionMnist_mean.npy')
        cova = np.load('./results/statistical_value/lenet_fashionMnist_cova.npy')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    fmd = FMD(undef_victim, args.batch_size, args.device, args.dims, mean, cova ,args.num_workers)
    # victim = PatchPlugin(victim,args.batch_size,args.device,args.dims,mean,cova,args.num_workers)

    max_fid_value = 0
    for i , (imgs,_) in enumerate(dataloader):
        fid_value,_ = fmd.calculate_fid(imgs.cuda())
        if fid_value>max_fid_value:
            max_fid_value = fid_value
        print('FMD: {} Max FMD{}'.format(fid_value,max_fid_value))


if __name__ == '__main__':
    main()
