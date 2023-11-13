#########################
#######Undefended########
#########################
# Undef MNIST-soft
CUDA_VISIBLE_DEVICES=0 python attack.py --dims 120  --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/lenet_mnist.pth --dataset_tar mnist --dataset_sur fashion  --model_tar lenet --model_sur lenet --epochs 50
# Undef MNIST-hard
CUDA_VISIBLE_DEVICES=0 python attack.py --dims 120  --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/lenet_mnist.pth --dataset_tar mnist --dataset_sur fashion  --model_tar lenet --model_sur lenet --epochs 50

##Undef FashionMNIST-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 120 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/lenet_fashion.pth --dataset_tar fashion --dataset_sur mnist  --model_tar lenet --model_sur lenet --epochs 50
##Undef FashionMNIST-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 120  --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/lenet_fashion.pth --dataset_tar fashion --dataset_sur mnist  --model_tar lenet --model_sur lenet --epochs 50
#
##Undef CIFAR10-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 256 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/wrn16_4_cifar10.pth --dataset_tar cifar10 --dataset_sur cifar100  --model_tar wres16 --model_sur wres16 --epochs 50
##Undef CIFAR10-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 256 --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/wrn16_4_cifar10.pth --dataset_tar cifar10 --dataset_sur cifar100  --model_tar wres16 --model_sur wres16 --epochs 50
#
##Undef CIFAR100-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 256 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/wrn16_4_cifar100.pth --dataset_tar cifar100 --dataset_sur cifar10  --model_tar wres16 --model_sur wres16 --epochs 50
##Undef CIFAR100-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --dims 256 --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/wrn16_4_cifar100.pth --dataset_tar cifar100 --dataset_sur cifar10  --model_tar wres16 --model_sur wres16 --epochs 50

#########################
######SFD Defended#######
#########################
# SFD_Ddf MNIST-soft
CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 1500 --window_size 64 --query_batch_size 1 --dims 120  --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/lenet_mnist.pth --dataset_tar mnist --dataset_sur fashion  --model_tar lenet --model_sur lenet --epochs 50
# SFD_Ddf MNIST-hard
CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 1500 --window_size 64 --query_batch_size 1 --dims 120  --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/lenet_mnist.pth --dataset_tar mnist --dataset_sur fashion  --model_tar lenet --model_sur lenet --epochs 50

##SFD_Ddf FashionMNIST-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 150 --window_size 64 --query_batch_size 1 --dims 120 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/lenet_fashion.pth --dataset_tar fashion --dataset_sur mnist  --model_tar lenet --model_sur lenet --epochs 50
##SFD_Ddf FashionMNIST-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 150 --window_size 64 --query_batch_size 1 --dims 120  --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/lenet_fashion.pth --dataset_tar fashion --dataset_sur mnist  --model_tar lenet --model_sur lenet --epochs 50
#
##SFD_Ddf CIFAR10-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 5 --window_size 64 --query_batch_size 1 --dims 256 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/wrn16_4_cifar10.pth --dataset_tar cifar10 --dataset_sur cifar100  --model_tar wres16 --model_sur wres16 --epochs 50
##SFD_Ddf CIFAR10-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 5 --window_size 64 --query_batch_size 1 --dims 256 --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/wrn16_4_cifar10.pth --dataset_tar cifar10 --dataset_sur cifar100  --model_tar wres16 --model_sur wres16 --epochs 50
#
##SFD_Ddf CIFAR100-soft
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 50 --window_size 64 --query_batch_size 1 --dims 256 --clone_batch_size 128 --pred_type soft --victim_dir ../pretrained/wrn16_4_cifar100.pth --dataset_tar cifar100 --dataset_sur cifar10  --model_tar wres16 --model_sur wres16 --epochs 50
##SFD_Ddf CIFAR100-hard
#CUDA_VISIBLE_DEVICES=0 python attack.py --defence --inter_thre 50 --window_size 64 --query_batch_size 1 --dims 256 --clone_batch_size 128 --pred_type hard --victim_dir ../pretrained/wrn16_4_cifar100.pth --dataset_tar cifar100 --dataset_sur cifar10  --model_tar wres16 --model_sur wres16 --epochs 50

