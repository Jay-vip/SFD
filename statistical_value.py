import os
import os
import pathlib
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
# from classifiers.resnet import resnet34,resnet18

from networks.lenet import LeNet5
from networks.wresnet import wrn_16_4
from networks.resnet import resnet18
from torchvision.datasets import CIFAR10,CIFAR100,MNIST,FashionMNIST

import numpy as np
import torchvision
import gc
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import torchvision.transforms as TF
from scipy import linalg
from torchvision import transforms
from attacks.dataset import get_dataloaders

try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

# from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size to use')
parser.add_argument('--num-workers', type=int,default=8,
                    help=('Number of processes to use for data loading. '
                          'Defaults to `min(8, num_cpus)`'))
parser.add_argument('--device', type=str, default=0,
                    help='Device to use. Like cuda, cuda:0 or cpu')
parser.add_argument('--dims', type=int, default=512,
                    choices=[1024,256,512,120,2048],
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('--model_path', type=str,default='./pretrained/resnet18_flowers17.pth',
                    help=('Paths to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--num_class', type=int, default=17)
parser.add_argument('--dataset', type=str, default='./datasets/')
parser.add_argument('--save_path', type=str, default='./statistical_value/')
args, unparsed = parser.parse_known_args()



IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

# model = load_teacher(args.model_path, 10)


def load_victim():

    if 'cifar10' in args.model_path or  'cifar100' in args.model_path :
        victim = wrn_16_4(args.num_class).cuda()
    if 'lenet' in args.model_path :
        victim = LeNet5().cuda()
    if 'resnet18' in args.model_path :
        victim = resnet18(num_classes=args.num_class).cuda()

    victim.load_state_dict(torch.load(args.model_path, map_location='cpu')['net'])
    victim.eval()

    return victim

class FMD():

    def __init__(self,model,batch_size,device,dims,mean,cova ):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.mean = mean
        self.cova = cova


    def get_activations(self,images_data):
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
        self.model.eval()

        pred_arr = np.empty((len(images_data)*self.batch_size, self.dims))
        start_idx = 0

        length = len(images_data)

        for i in range(length):
            batch = images_data[i].to(self.device)
            with torch.no_grad():
                feat = self.model(batch,return_rep=True)[1]

            pred_arr[start_idx:start_idx + feat.shape[0]] = feat.cpu()
            start_idx = start_idx + feat.shape[0]

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
            print(msg)
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


    def calculate_activation_statistics(self,images_data):
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
        act = self.get_activations(images_data)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma


def main():

    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.2, 0.2, 0.2))])

    transform_one = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    victim = load_victim()
    fmd = FMD(victim, args.batch_size, args.device, args.dims, None, None )



    if 'cifar10' in args.model_path:
        dataset = CIFAR10(root=args.dataset, train=True,
                                               download=True, transform=transform)
    if 'cifar100' in args.model_path:
        dataset = CIFAR100(root=args.dataset, train=True,
                                               download=True, transform=transform)
    if 'mnist' in  args.model_path:
        dataset = MNIST(root=args.dataset , train=True ,
                                               download=True ,transform=transform_one)
    if 'fashionMnist' in args.model_path:
        dataset = FashionMNIST(root=args.dataset , train=True,
                                               download=True , transform=transform_one)
    if 'resnet18' in args.model_path:
        dataloader,_ = get_dataloaders(ds='flowers17', batch_size=args.batch_size)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)




    img_data = []
    for i,(imgs,labs) in enumerate(dataloader):
        imgs = imgs.cuda()
        img_data.append(imgs)
    mu,sig = fmd.calculate_activation_statistics(img_data)

    np.save(args.save_path+args.model_path.split('/')[-1][:-4]+'_mean.npy', mu)
    np.save(args.save_path+args.model_path.split('/')[-1][:-4]+'_cova.npy', sig)

if __name__ == '__main__':
    main()
