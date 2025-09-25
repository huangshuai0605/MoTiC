import argparse
import importlib
from utils import *

MODEL_DIR= None
DATA_DIR = '/workspace/huangshuai/CEC-CVPR2021/data/'
PROJECT='clip_distill'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-epochs_new', type=int, default=100)
    
    parser.add_argument('-base_mode', type=str, default='avg_cos',   #base阶段mode
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',   #incremental阶段mode
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier
    
    parser.add_argument('-lr_new', type=float, default=0.1)         #new  session的学习率
    parser.add_argument('-decay', type=float, default=0.0005)       #权重衰减系数，防止过拟合，类似于L2正则化，大的话会欠拟合，小的话会过拟合
    
    parser.add_argument('-data_init', default=True,action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    
    parser.add_argument('-batch_size_base', type=int, default=256)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-gpu', type=int, default='1')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-num_workers', type=int, default=8)
    
    parser.add_argument('-mlp_hidden_size', type=int, default=64)
    parser.add_argument('-temperature', type=int, default=16, help='the distill temperature')
    parser.add_argument('-alpha', type=int, default=10000,help='the distill hyper-parameters')
    parser.add_argument('-sample_num', type=int, default=5,help='sample nums')
    
    parser.add_argument('--num_aug', type=int, default=0)
    return parser

if __name__ == '__main__':
    parser = get_command_line_parser()          #获取参数
    args = parser.parse_args()                  #解析参数
    set_seed(args.seed)                         #随机种子,实验可重复性
    pprint(vars(args))
    #args.num_gpu = set_gpu(args)                #设置可用gpu 

    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train() 