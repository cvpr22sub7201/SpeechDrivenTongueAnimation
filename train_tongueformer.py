import argparse
from io import open
import random
import time
import os.path as osp
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from easydict import EasyDict as edict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TongueMocapDataset
from models import TongueFormer, save_checkpoint
from losses import L2L1Loss, HuberLoss, ShrinkageLoss
from logger.model_logger import ModelLogger


random.seed(78373)
torch.manual_seed(78373)
np.random.seed(78373)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local_config_path', type=str,
                        help='Path to the local machine configuration')
    parser.add_argument('-c', '--config_path', type=str,
                        help='Path to the experiment configuration')
    parser.add_argument('-gid', '--gpu_id', type=int, default=0,
                        help='GPU to be used for training')
    return parser.parse_args()


def print_training_header(training):
    print('\nTraining Model')
    print('=' * 70)
    print(f'Num Epochs: {training.num_epochs}\nbatch size: {training.batch_sz}')
    print(f'Loss: {training.loss.label}  Params: {training.loss.params}')
    print('=' * 70)
    print()


def get_criterion(loss):
    """Creates a criterion from a label for MSE, Huber, and L1, otherwise None
    Args:
        loss (dict): dictionary with the loss label and params to construct the criterion
    Returns:
        nn.criterion: pytorch loss calculator
    """
    if loss.label == 'mse':
        return nn.MSELoss()
    elif loss.label == 'huber':
        return HuberLoss(delta=loss.params.delta)
    elif loss.label == 'smooth_l1':
        return nn.SmoothL1Loss(beta=loss.params.beta)
    elif loss.label == 'l1':
        return nn.L1Loss()
    elif loss.label == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss.label == 'l2l1':
        return L2L1Loss(betas=loss.params.betas)
    elif loss.label == 'shrinkage':
        return ShrinkageLoss(speed=loss.params.speed, loc=loss.params.loc)
    
    return None


def get_optimizer(model, optim):
    """Creates an optimizer from a label, if not in the list returns None
    Args:
        optim (dict): dictionary with the params to construct the optimizer
    Returns:
        nn.optim: pytorch optimizer
    """
    if optim.label == 'adam':
        return torch.optim.Adam(model.parameters(), lr=optim.params.lr, weight_decay=optim.params.weight_decay)
    if optim.label == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=optim.params.lr, weight_decay=optim.params.weight_decay)
    
    return None


def get_scheduler(scheduler, optimizer):
    """Creates an optimizer from a label, if not in the list returns None
    Args:
        optim (dict): dictionary with the params to construct the optimizer
    Returns:
        nn.optim: pytorch optimizer
    """
    if scheduler.label == 'exp_lr':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=scheduler.gamma)

    return None


def train(model, model_optimizer, criterion, scheduler, dataloaders, save_dir,
            model_logger, log_start_step,
            batch_size, n_epochs, device, optim,
            output_idx, output_full=False,
            start_epoch=1, early_stop=10,
            multi_gpu=False,
            print_every=200):

    best_score = edict(value=1e9, epoch=0)
    iter_times_list = list()
    global_step = log_start_step
    last_val_loss = 1e9
    for epoch in range(start_epoch, start_epoch+n_epochs):
        print(f'Epoch {epoch}/{start_epoch+n_epochs-1}')
        print('-' * 70)
        
        ###------ Train ------###
        print('Training phase')
        phase = 'train'
        model.train(True)
        epoch_start_time = time.time()
        iter_start_time = time.time()

        dataloader_iter = iter(dataloaders['train'])
        running_loss = 0.0
        batch_idx = 0
        train_loss = 0.0
        for source_tensor, target_pos_tensor, _ in dataloader_iter:
            # skip the batch that is not complete
            if source_tensor.shape[0] != batch_size:
                continue
            
            global_step += 1
            batch_idx += 1

            # Pass to device
            source_tensor = source_tensor.to(device).float()            
            output_tensor = target_pos_tensor if output_full else target_pos_tensor[:, output_idx, :]

            pos_pred = model(source_tensor)

            loss = criterion(pos_pred, output_tensor.to(device).float())
            if batch_idx % print_every == 0:
                model_logger.train.add_scalar('loss/iter', loss, global_step)
            
            # We can perfectly fit a batch in a single pass
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % print_every == 0:
                iter_time = time.time() - iter_start_time
                iter_times_list.append(iter_time)
                iter_start_time = time.time()
                print(f'[train] Epoch {epoch}  Iter Time:{iter_time:.3f}  Step:{batch_idx}/{len(dataloaders["train"]):<8}  l: {running_loss/batch_idx:<10}')
        
        train_loss = running_loss/len(dataloaders[phase])
        epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_start_time))
        print(f'Training totals:  loss: {train_loss}  time: {epoch_time_str}')
        print()

        ###------ Validate every 10 epochs ------###
        if epoch % 10 == 0:
            print('Validation phase')
            phase = 'valid'
            model.eval()
        
            with torch.no_grad():
                running_loss = 0.0
                epoch_start_time = time.time()
                iter_start_time = time.time()
                
                dataloader_iter = iter(dataloaders['valid'])
                running_loss = 0.0
                batch_idx = 0
                val_loss = 0.0
                for source_tensor, target_pos_tensor, _ in dataloader_iter:
                    # skip the batch that is not complete
                    if source_tensor.shape[0] != batch_size:
                        continue

                    batch_idx += 1

                    # Pass to device
                    source_tensor = source_tensor.to(device).float()
                    output_tensor = target_pos_tensor if output_full else target_pos_tensor[:, output_idx, :]

                    pos_pred = model(source_tensor)
                    loss = criterion(pos_pred, output_tensor.to(device).float())
                    running_loss += loss.item()
                    
                    if batch_idx % print_every == 0:
                        iter_time = time.time() - iter_start_time
                        iter_times_list.append(iter_time)
                        iter_start_time = time.time()
                        print(f'[valid] Epoch {epoch}  Iter Time:{iter_time:.3f}  Step:{batch_idx}/{len(dataloaders["valid"]):<8}  l: {running_loss/batch_idx:<10}')
                
                val_loss = running_loss/len(dataloaders['valid'])
                epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_start_time))
                print(f'Validation totals:  loss: {val_loss}  time: {epoch_time_str}')
                print()
                model_logger.val.add_scalar('loss/iter', val_loss, global_step)

                # -- Early Stop ---
                if val_loss < best_score.value:
                    best_score.value = val_loss
                    best_score.epoch = epoch
                
                if (epoch - best_score.epoch) >= early_stop:
                    print(f'Early stop at epoch {epoch}, previous best: {best_score.value} @ {best_score.epoch}')
                    break

                last_val_loss = val_loss

        # TODO: HACK! add as a config parameter
        if (last_val_loss < 2.0) or (epoch % 10 == 0):
            save_checkpoint(epoch=epoch, 
                            model=model, 
                            model_params=model.module.params if multi_gpu else model.params, 
                            optimizer=model_optimizer, 
                            optimizer_params=dict(optim.params), 
                            loss=train_loss, 
                            global_step=global_step,
                            save_path=osp.join(save_dir, f'{epoch:02d}.pt'))
        
        if scheduler is not None:
            scheduler.step()
        
    total_train_time_str = time.strftime("%H:%M:%S", time.gmtime(sum(iter_times_list)))
    print(f'Total training time: {total_train_time_str}')
    return train_loss, val_loss


def main(local_config, config):
    # Training params
    start_epoch = 1
    
    # Data loading params
    train_dataset_path = osp.join(local_config.datasets_dir, config.data.train.path)
    valid_dataset_path = osp.join(local_config.datasets_dir, config.data.valid.path)

    # Training save dir
    model_save_dir = osp.join(local_config.models_dir, config.model.save_dir)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    # Log
    log_dir = osp.join(local_config.logs_dir, config.log.save_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_start_step = 0
    model_logger = ModelLogger(log_dir, config.name)
    
    # Create device under which the model will be trained
    device_str = 'cpu'
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        target_gid = int(args.gpu_id)
        if target_gid >= 0 and target_gid < num_gpus:
            device_str = f'cuda:{target_gid}'
    
    device = torch.device(device_str)
    print(f'Training on device: {device}')
    
    # Create dataset
    print('Loading Training data')
    train_dataset = TongueMocapDataset(train_dataset_path,
                                        num_files=config.data.train.num_files,
                                        win_sz=config.data.train.win_sz,
                                        stride=config.data.train.win_stride,
                                        pose_only=config.data.train.pose_only)
    print(f'Training samples:    {len(train_dataset)}')
    
    print('Loading Validation data')
    valid_dataset = TongueMocapDataset(valid_dataset_path,
                                        num_files=config.data.valid.num_files,
                                        win_sz=config.data.valid.win_sz,
                                        stride=config.data.valid.win_stride,
                                        pose_only=config.data.valid.pose_only)
    print(f'Validation samples:  {len(valid_dataset)}')
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, 
                                    batch_size=config.training.batch_sz, 
                                    shuffle=True, 
                                    num_workers=config.data.train.num_workers, 
                                    pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, 
                                    batch_size=config.training.batch_sz, 
                                    shuffle=False, 
                                    num_workers=config.data.valid.num_workers, 
                                    pin_memory=True)
    dataloaders = dict(train=train_dataloader, 
                        valid=valid_dataloader)

    # Build if no checkpoint is given
    if 'checkpoint' not in config:
        print('Building new model')
        model = TongueFormer(**config.model.params)
        model.to(device)

        print('Building new optimizer')
        model_optimizer = get_optimizer(model, config.optim)
    else:
        print('Loading model checkpoint')
        checkpoint_path = osp.join(local_config.models_dir, config.checkpoint.path)
        checkpoint = torch.load(checkpoint_path)
        model = TongueFormer(**config.model.params)
        trained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        diff = set(model_dict.keys()) - set(trained_dict.keys())
        froce_reset_optim = False
        if not diff:
            model.load_state_dict(trained_dict)
        else:
            if 'deeper_fc' in config.model.params:
                if config.model.params['deeper_fc']:
                    froce_reset_optim = True
                    # let's remove head.1 since it's size is now
                    # different and not matching the checkpoint
                    for i in range(2):
                        del trained_dict['head.{}.weight'.format(i)]
                        del trained_dict['head.{}.bias'.format(i)]

            model_dict.update(trained_dict)
            model.load_state_dict(model_dict)

        model.to(device)

        reset = False
        if 'reset' in config.checkpoint:
            reset = config.checkpoint.reset

        if not (reset or froce_reset_optim): 
            print('Loading optimizer checkpoint')
            model_optimizer = optim.Adam(model.parameters(), lr=checkpoint['optimizer_params']['lr'])
            model_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = int(Path(checkpoint_path).stem) + 1
            log_start_step = checkpoint['global_step']
        else:
            print('RESET is TRUE - Building new optimizer')
            model_optimizer = get_optimizer(model, config.optim)

    
    # Criterion
    criterion = get_criterion(config.training.loss)
    criterion.to(device)

    # Scheduler
    scheduler = get_scheduler(config.scheduler, model_optimizer) if 'scheduler' in config else None

    print_training_header(config.training)
    if start_epoch > 1:
        print(f'Resuming training from epoch {start_epoch}')
    print(f'Checkpoints save dir: {model_save_dir}')
    print(f'Log save dir:         {log_dir}')
    print()
    
    train(model=model, 
            model_optimizer=model_optimizer,
            criterion=criterion,
            scheduler=scheduler,
            dataloaders=dataloaders, 
            save_dir=model_save_dir, 
            model_logger=model_logger,
            log_start_step=log_start_step,
            batch_size=config.training.batch_sz, 
            n_epochs=config.training.num_epochs, 
            device=device,
            optim=config.optim,
            output_idx=config.training.output_idx,
            output_full=config.training.output_full, # TODO: add to all the tongueformer training config files
            start_epoch=start_epoch,
            early_stop=config.training.early_stop,
            multi_gpu=config.model.multi_gpu,
            print_every=200)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_args()
    # Load configurations
    yaml = YAML(typ='safe')
    # -- machine configuration
    local_config = edict(yaml.load(open(args.local_config_path)))
    # -- training configuration
    config = edict(yaml.load(open(args.config_path)))

    # TODO: fix None load
    config.model.params.qk_scale = None
    config.model.params.norm_layer = None
    
    main(local_config, config)