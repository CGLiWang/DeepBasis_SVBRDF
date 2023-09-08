import argparse
import torch
from os import path as osp
import os
import setproctitle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data.dataset import DeepBasisDataset
from model.deepbasis_model import DeepBasisModel
from utils import get_root_logger, get_root_logger,MessageLogger
import logging


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='experiment name.')
    parser.add_argument("--mode",type=str,default='real',help="train / test / real")
    parser.add_argument('--save_root',type=str,required=True,help="root path to save results.")
    parser.add_argument('--real_data_root',type=str,required=True,help="root path for data.")
    parser.add_argument('--loadpath_network_g',type=str,required=True)
    parser.add_argument('--loadpath_network_l',type=str,required=True)
    parser.add_argument('--fovZ',type=int,default=2.414)
    
    
    args = parser.parse_args()


    args.save_root = osp.join(args.save_root,"real",args.name)
    makedirs(args.save_root)


    return args


def create_dataloader(args):
      
    dataset_opt = {
        'name': 'DeepBasisDataset',
        'svbrdf_root': args.real_data_root,
        'log': True,
        'real_input': True,
        'input_gamma': True
    }
    test_set = DeepBasisDataset(dataset_opt)
    dataloader_opt = {
        'dataset': test_set,
        'batch_size': 1,
        'shuffle': False,
        'pin_memory': True,
        'num_workers': 1,
    }
    test_loader = torch.utils.data.DataLoader(**dataloader_opt)
    return test_loader

def init_loggers(args):
    log_file = osp.join(args.save_root, f"test_{args.name}.log")
    logger = get_root_logger(logger_name='DeepBasis', log_level=logging.INFO, log_file=log_file)
    args_str = ""
    for arg_name, arg_value in vars(args).items():
        args_str += f"{arg_name}: {arg_value}\r\n "
    logger.info(args_str)
    return logger

def test_pipeline(args):
    torch.backends.cudnn.benchmark = True
    
    # initialize loggers
    logger = init_loggers(args)

    # create train and validation dataloaders
    test_loader = create_dataloader(args)

    # create model
    model = DeepBasisModel(args)

    model.validation(test_loader, "test")
    log_str = f'\t # pixel: {model.metric_results:.4f}\t'
    logger.info(log_str)
        


        

if __name__ == '__main__':
    proc_title = "DeepBasis_test"
    setproctitle.setproctitle(proc_title)

    args = parse_options()
    
    test_pipeline(args)
