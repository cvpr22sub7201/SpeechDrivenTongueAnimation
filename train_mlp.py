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
from torch.utils.data import DataLoader

from dataset import CausalTongueMocapDataset
from models import MLP, save_checkpoint, load_checkpoint
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


def print_training_header(learning_rate, dropouts, batch_size, model, model_save_dir, log_dir):
    print('\nTraining Model')
    print('=' * 70)
    print(f'LR:{learning_rate}\ndropout: {dropouts}\nbatch size: {batch_size}\nmodel:{model}\nmodel dir:{model_save_dir}\nlog dir:{log_dir}')
    print('=' * 70)
    print()


def print_checkpoint(name, params_dict):
    print(f'{"Name":>13}: {name}')
    for k, v in params_dict.items():
        print(f'{k:>13}: {v}')


def get_criterion(loss_label):
    """Creates a criterion from a label for MSE, Huber, and L1, otherwise None

    Args:
        loss_label (str): loss label string

    Returns:
        nn.criterion: pytorch loss calculator
    """
    if loss_label == 'mse':
        return nn.MSELoss()
    elif loss_label == 'huber':
        return nn.SmoothL1Loss()
    elif loss_label == 'l1':
        return nn.L1Loss()

    return None


def train(model, model_optimizer, dataloaders, save_dir, log_dir,
          batch_size, n_epochs, device, start_epoch=1, multi_gpu=False, learning_rate=0.01,
          loss_str='mse', print_every=200, log_start_step=0, label='training'):

    model_logger = ModelLogger(log_dir, label)
    criterion = get_criterion(loss_str)
    criterion.to(device)

    iter_times_list = list()
    global_step = log_start_step
    for epoch in range(start_epoch, start_epoch+n_epochs):
        print(f'Epoch {epoch}/{start_epoch+n_epochs-1}')
        print('-' * 70)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                print('Training phase')
            else:
                model.eval()
                print('Validation phase')

            dataloader_iter = iter(dataloaders[phase])
            running_loss = 0.

            epoch_start_time = time.time()
            iter_start_time = time.time()

            batch_idx = 0
            total_loss = 0.0
            for feats_tensor, pos_tensor, _ in dataloader_iter:
                # skip the batch that is not complete
                if feats_tensor.shape[0] != batch_size:
                    continue

                # Flatten for causal inference
                input_tensor = torch.flatten(feats_tensor, 1, 2).to(device).float()
                target_pos_tensor = torch.flatten(pos_tensor, 1, 2).to(device).float()

                pos_pred = model(x=input_tensor)
                loss = criterion(pos_pred, target_pos_tensor)

                # We can perfectly fit a batch in a single pass
                model_optimizer.zero_grad()
                if phase == 'train':
                    if (batch_idx > 0) and (batch_idx % print_every == 0):
                        model_logger.train.add_scalar(
                            'loss/iter', loss, global_step)
                    loss.backward()
                    model_optimizer.step()

                running_loss += loss.item()

                if (batch_idx > 0) and (batch_idx % print_every == 0):
                    iter_time = time.time() - iter_start_time
                    iter_times_list.append(iter_time)
                    iter_start_time = time.time()
                    print(f'[{phase}] Epoch {epoch}  Iter Time:{iter_time:.3f}  Step:{batch_idx}/{len(dataloaders[phase]):<8}  l: {running_loss/batch_idx:<10}')

                batch_idx += 1
                if phase == 'train':
                    global_step += 1

            total_loss = running_loss/len(dataloaders[phase])
            epoch_time_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - epoch_start_time))
            print(f'Totals: {phase}  loss: {total_loss}  time: {epoch_time_str}')
            print()

            if phase == 'train':
                save_checkpoint(epoch=epoch,
                                model=model,
                                model_params=model.module.params if multi_gpu else model.params,
                                optimizer=model_optimizer,
                                optimizer_params=dict(lr=learning_rate),
                                loss=total_loss,
                                global_step=global_step,
                                save_path=osp.join(save_dir, f'{epoch:02d}.pt'))
            else:
                model_logger.val.add_scalar(
                    'loss/iter', total_loss, global_step)

    total_train_time_str = time.strftime("%H:%M:%S", time.gmtime(sum(iter_times_list)))
    print(f'Total training time: {total_train_time_str}')


def main(local_config, config):
    # Training params
    start_epoch = 1
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_sz
    dropouts = config.training.dropouts
    learning_rate = config.training.learning_rate

    # Data loading params
    num_audio_files = config.data.num_train_files
    dataload_workers = config.data.num_dataload_workers
    data_input_win_sz = config.data.input_win_sz
    data_output_win_sz = config.data.output_win_sz
    data_win_stride = config.data.win_stride
    train_dataset_path = osp.join(local_config.datasets_dir, config.data.train_path)
    valid_dataset_path = osp.join(local_config.datasets_dir, config.data.valid_path)

    # Model params
    input_size = config.model.input_sz
    hidden_sizes = config.model.hidden_szs
    output_size = config.model.output_sz
    model_save_dir = osp.join(local_config.models_dir, config.model.save_dir)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    # Log params
    log_dir = osp.join(local_config.logs_dir, config.log.save_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_start_step = 0

    # Create device under which the model will be trained
    device_str = 'cpu'
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if config.training.multi_gpu and num_gpus > 1:
            device_str = 'cuda'
        else:
            target_gid = int(args.gpu_id)
            if target_gid >= 0 and target_gid < num_gpus:
                device_str = f'cuda:{target_gid}'

    print(f'Training on device: {device_str}')
    device = torch.device(device_str)

    # Create dataset
    print('Loading Training data')
    train_dataset = CausalTongueMocapDataset(train_dataset_path,
                                             num_files=num_audio_files,
                                             input_win_sz=data_input_win_sz,
                                             output_win_sz=data_output_win_sz,
                                             stride=data_win_stride,
                                             pose_only=False)
    print(f'Training samples:    {len(train_dataset)}')

    print('Loading Validation data')
    valid_dataset = CausalTongueMocapDataset(valid_dataset_path,
                                             num_files=num_audio_files,
                                             input_win_sz=data_input_win_sz,
                                             output_win_sz=data_output_win_sz,
                                             stride=data_win_stride,
                                             pose_only=False)
    print(f'Validation samples:  {len(valid_dataset)}')

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=dataload_workers, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=dataload_workers, pin_memory=True)
    dataloaders = dict(train=train_dataloader,
                       valid=valid_dataloader)

    # Build if no checkpoint is given
    model = None
    if 'checkpoint' not in config:
        print('Building new model')
        model = MLP(input_dim=input_size,
                    hidden_dims=hidden_sizes,
                    output_dim=output_size,
                    dropouts=dropouts)
        if config.training.multi_gpu:
            model = nn.DataParallel(model)

        print('Building new optimizer')
        model_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print('Loading checkpoint')
        checkpoint_path = osp.join(local_config.models_dir, config.checkpoint.path)
        model, model_optimizer, log_start_step = load_checkpoint(checkpoint_path=checkpoint_path,
                                                                 model_class=MLP,
                                                                 optimizer_class=optim.Adam,
                                                                 device=device)
        start_epoch = int(Path(checkpoint_path).stem) + 1

    model.to(device)

    print_training_header(learning_rate, dropouts,
                          batch_size, model, model_save_dir, log_dir)

    train(model, model_optimizer,
          dataloaders=dataloaders,
          save_dir=model_save_dir, log_dir=log_dir,
          batch_size=batch_size, learning_rate=learning_rate,
          start_epoch=start_epoch, n_epochs=num_epochs,
          loss_str=config.training.loss,
          device=device,
          multi_gpu=config.training.multi_gpu,
          log_start_step=log_start_step,
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

    main(local_config, config)
