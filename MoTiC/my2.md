#cub200
python -m pdb train.py  -dataset cub200 -lr_base 0.005 -epochs_base 70 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.01 --inter_lamb 2.0 --ssc_temp 0.07 --temp 32 -project base5_add

#cub200 diffusion
python -m pdb train.py  -dataset cub200 -lr_base 0.005 -epochs_base 70 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.01 --inter_lamb 2.0 --ssc_temp 0.07 --temp 32 -project base5_add_diffusion --diffusion_basetrainset_dir /workspace/huangshuai/diffuseMix/results/cub200/baseSession/blended/

#cifar100
python -m pdb train.py  -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 2.8 --ssc_temp 0.01 --temp 32 -project base5_add
#cifar100 diffusion
python -m pdb train.py  -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 2.8 --ssc_temp 0.01 --temp 32 -project base5_add_diffusion --diffusion_basetrainset_dir /workspace/huangshuai/diffuseMix/results/cifar100/baseSession/blended/

#cub200 rotation
python -m pdb train.py  -dataset cub200 -lr_base 0.005 -epochs_base 70 -gpu 0,1 -batch_size_base 64 -seed 1 --ssc_lamb 0.01 --inter_lamb 2.5 --ssc_temp 0.07 --temp 32 -project base5_add_rotation_classify


python -m pdb train.py  -dataset cub200 -lr_base 0.005 -epochs_base 70 -gpu 0,1 -batch_size_base 64 -seed 1 --ssc_lamb 0.01 --inter_lamb 2.5 --ssc_temp 0.07 --temp 32 -project base5_add_rotation4_classify -fantasy rotation2

#cifar100 ratation
python -m pdb train.py  -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 2.5 --ssc_temp 0.01 --temp 32 -project base5_add_rotation_classify 

python -m pdb train.py  -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 2.5 --ssc_temp 0.01 --temp 32 -project base5_add_rotation4_classify -fantasy rotation


#miniImageNet rotation
python -m pdb train.py  -dataset mini_imagenet -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 1.0 --ssc_temp 0.03 --temp 32 -project base5_add_rotation_classify

python -m pdb train.py  -dataset mini_imagenet -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 1.0 --ssc_temp 0.03 --temp 32 -project base5_add_rotation4_classify -fantasy rotation

#final
#cub200
python -m pdb train.py  -dataset cub200 -lr_base 0.005 -epochs_base 70 -gpu 0,1 -batch_size_base 64 -seed 1 --ssc_lamb 0.01 --inter_lamb 2.5 --ssc_temp 0.07 --temp 32 -project motic -fantasy rotation2

#cifar100 
python -m pdb train.py  -dataset cifar100 -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 2.5 --ssc_temp 0.01 --temp 32 -project motic -fantasy rotation2

#miniImageNet
python -m pdb train.py  -dataset mini_imagenet -lr_base 0.1 -epochs_base 200 -gpu 0,1 -batch_size_base 128 -seed 1 --ssc_lamb 0.1 --inter_lamb 1.5 --ssc_temp 0.03 --temp 32 -project motic -fantasy rotation
