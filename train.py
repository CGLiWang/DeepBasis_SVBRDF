import argparse
import datetime
import math
import random
import time
from functools import partial
import torch
from os import path as osp
import os
import setproctitle
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from data.data_sampler import EnlargedSampler
from data.dataset import DeepBasisDataset
from model.deepbasis_model import DeepBasisModel
from utils import get_root_logger, get_root_logger,MessageLogger
import logging

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help='experiment name.')
    parser.add_argument("--mode",type=str,default='train',help="train / test / real")
    parser.add_argument('--save_root',type=str,required=True,help="root path to save results.")
    parser.add_argument('--train_data_root',type=str,default="./dataset/train",help="root path for data.")
    parser.add_argument('--test_data_root',type=str,required=True,help="root path for data.")
    parser.add_argument('--dataset_enlarge_ratio',type=int,default=1)
    parser.add_argument('--bSize',type=int,default=4)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--nWorkers',type=int,default=8)
    parser.add_argument('--total_iter',type=int,default=1000)
    parser.add_argument('--weight_vc',type=float,default=0.05)
    parser.add_argument('--print_freq',type=int,default=1)
    parser.add_argument('--save_freq',type=int,default=50)
    parser.add_argument('--test_freq',type=int,default=50)
    parser.add_argument('--use_tb_logger',action='store_true')
    parser.add_argument('--fovZ',type=int,default=2.414)
    
    
    args = parser.parse_args()


    args.save_root = osp.join(args.save_root,"train",args.name)
    makedirs(args.save_root)

    

    # random seed
    seed = random.randint(1, 10000)
    args.seed = seed
    set_random_seed(seed)

    return args

def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def create_dataloader(args):
    
    # create train and test dataloaders
    dataset_enlarge_ratio = args.dataset_enlarge_ratio
    dataset_opt = {
        'name': 'DeepBasisDataset',
        'svbrdf_root': args.train_data_root,
        'log': True,
    }
    train_set = DeepBasisDataset(dataset_opt)
    train_sampler = EnlargedSampler(train_set, ratio=dataset_enlarge_ratio)
    
    dataloader_opt = {
        'dataset': train_set,
        'batch_size': args.bSize,
        'shuffle': False,
        'num_workers': args.nWorkers,
        'sampler': train_sampler,
        'drop_last': True,
        'pin_memory': True,
        'worker_init_fn': partial(worker_init_fn, num_workers=args.nWorkers, rank=0, seed=args.seed)
    }
    
    train_loader = torch.utils.data.DataLoader(**dataloader_opt)

    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio / (args.bSize))
    total_iters = args.total_iter
    total_epochs = math.ceil(total_iters / (num_iter_per_epoch))
    print('Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                f'\n\tBatch size per gpu: {args.bSize}'
                f'\n\tWorld size (gpu number): {1}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')
        
    dataset_opt = {
        'name': 'DeepBasisDataset',
        'svbrdf_root': args.test_data_root,
        'log': True,
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
    return train_loader, train_sampler, test_loader, total_epochs, total_iters


class CUDAPrefetcher():
    """CUDA prefetcher.

    Ref:
    https://github.com/NVIDIA/apex/issues/304#

    It may consums more GPU memory.

    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()

def init_loggers(args):
    log_file = osp.join(args.save_root, f"train_{args.name}.log")
    logger = get_root_logger(logger_name='DeepBasis', log_level=logging.INFO, log_file=log_file)
    args_str = ""
    for arg_name, arg_value in vars(args).items():
        args_str += f"{arg_name}: {arg_value}\r\n "
    logger.info(args_str)
    

    # initialize wandb logger before tensorboard logger to allow proper sync:

    tb_logger = None
    if  args.use_tb_logger:
        from torch.utils.tensorboard import SummaryWriter
        tb_logger = SummaryWriter(log_dir=osp.join(args.save_root,"tb_%s" % args.name))
    return logger, tb_logger


def train_pipeline(args):
    try:
        torch.backends.cudnn.benchmark = True
        
        # initialize loggers
        logger, tb_logger = init_loggers(args)

        # create train and validation dataloaders
        train_loader, train_sampler, test_loader, total_epochs, total_iters = create_dataloader(args)

        # create model
        model = DeepBasisModel(args)
        start_epoch = 0
        current_iter = 0

        # create message logger (formatted outputs)
        msg_logger = MessageLogger(args, current_iter, tb_logger)

        # dataloader prefetcher
        prefetcher = CUDAPrefetcher(train_loader)



        # training
        logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
        data_time, iter_time = time.time(), time.time()
        start_time = time.time()

        for epoch in range(start_epoch, total_epochs + 1):
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()
            

            while train_data is not None:
                start = time.time()
                data_time = time.time() - data_time

                current_iter += 1
                if current_iter > total_iters:
                    break
                # update learning rate
                model.update_learning_rate(current_iter)
                # training
                train_data['iter'] = current_iter
                model.feed_data(train_data)
                # print(current_iter, torch.cuda.memory_reserved()+torch.cuda.memory_allocated(), train_data['trace'].shape[1], torch.cuda.max_memory_allocated())
                model.optimize_parameters(current_iter)
                iter_time = time.time() - iter_time
                # log
                if current_iter % args.print_freq == 0:
                    log_vars = {'epoch': epoch, 'iter': current_iter}
                    log_vars.update({'lrs': model.get_current_learning_rate()})
                    log_vars.update({'time': iter_time, 'data_time': data_time})
                    log_vars.update(model.get_current_log())
                    msg_logger(log_vars)

                # save models and training states
                if current_iter % args.save_freq == 0:
                    logger.info('Saving models and training states.')
                    model.save(current_iter)

                # validation
                if current_iter % args.test_freq == 0:
                    model.validation(test_loader, current_iter)
                    log_str = f'\t # pixel: {model.metric_results:.4f}\t'
                    logger.info(log_str)
                    if tb_logger:
                        tb_logger.add_scalar(f'metrics/pixel', model.metric_results, current_iter)

                data_time = time.time()
                iter_time = time.time()
                train_data = prefetcher.next()
                # print("iteration time:",time.time()-start)
                # print("----------------iteration-------------------------")
            # end of iter

        # end of epoch
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        

        logger.info(f'End of training. Time consumed: {consumed_time}')
        logger.info('Save the latest model.')
        model.save(current_iter=-1)  # -1 stands for the latest
        model.validation(test_loader, current_iter, tb_logger)
        log_str = f'\t # pixel: {model.metric_results:.4f}\t'
        logger.info(log_str)
        if tb_logger:
            tb_logger.add_scalar(f'metrics/pixel', model.metric_results, current_iter)

        if tb_logger:
            tb_logger.close()
    except KeyboardInterrupt as e:
        consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
        logger.info(f'Exception at iteration: {current_iter}, epoch:{epoch}. Time consumed: {consumed_time}')
        logger.info('Save the model.')
        model.validation(test_loader, current_iter)
        log_str = f'\t # pixel: {model.metric_results:.4f}\t'
        logger.info(log_str)
        if tb_logger:
            tb_logger.add_scalar(f'metrics/pixel', model.metric_results, current_iter)
        model.save(current_iter)


        

if __name__ == '__main__':
    proc_title = "DeepBasis_train"
    setproctitle.setproctitle(proc_title)

    args = parse_options()
    
    train_pipeline(args)
