import argparse
import os
import os.path as osp
import random
import time
from io import open
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from torch import optim
from torch.utils.data import DataLoader

from dataset import  TongueMocapDataset
from logger.model_logger import ModelLogger
from models import GRU, save_checkpoint

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


def print_training_header(learning_rate, dropout_rate, batch_size, rnn, model_save_dir, log_dir):
    print('\nTraining Model')
    print('=' * 70)
    print(f'LR:{learning_rate}\ndropout: {dropout_rate}\nbatch size: {batch_size}\nmodel:{rnn}\nmodel dir:{model_save_dir}\nlog dir:{log_dir}')
    print('=' * 70)
    print()


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


def train(gru, gru_optimizer, dataloaders, save_dir, log_dir,
          batch_size, n_epochs, device, start_epoch=1, early_stop=10,
          multi_gpu=False, learning_rate=0.01, dropout=0.2,
          loss_str='mse', log_start_step=0, print_every=200):

    model_logger = ModelLogger(log_dir, f'test')
    criterion = get_criterion(loss_str)
    criterion.to(device)

    iter_times_list = list()

    best = edict(loss=1e9, epoch=-1, path='best.pt')
    global_step = log_start_step
    for epoch in range(start_epoch, start_epoch+n_epochs):
        print(f'Epoch {epoch}/{start_epoch+n_epochs-1}')
        print('-' * 70)

        for phase in ['train', 'valid']:
            if phase == 'train':
                gru.train(True)
                print('Training phase')
            else:
                gru.eval()
                print('Validation phase')

            epoch_start_time = time.time()
            iter_start_time = time.time()

            if not multi_gpu:
                h0 = gru.init_hidden(batch_size).to(device)

            dataloader_iter = iter(dataloaders[phase])
            running_loss = 0.
            batch_idx = 0
            total_loss = 0.0
            for input_tensor, target_tensor in dataloader_iter:
                # skip the batch that is not complete
                if input_tensor.shape[0] != batch_size:
                    continue

                if phase == 'train':
                    global_step += 1
                batch_idx += 1

                if multi_gpu:
                    pos_pred = gru(x=input_tensor.to(device).float())
                else:
                    h = h0.data
                    pos_pred, _ = gru(x=input_tensor.to(device).float(), h=h)

                loss = criterion(pos_pred, target_tensor.to(device).float())

                # We can perfectly fit a batch in a single pass
                gru_optimizer.zero_grad()
                if phase == 'train':
                    if batch_idx % print_every == 0:
                        model_logger.train.add_scalar('loss/iter',
                                                      loss,
                                                      global_step)
                    loss.backward()
                    gru_optimizer.step()

                running_loss += loss.item()

                if batch_idx % print_every == 0:
                    iter_time = time.time() - iter_start_time
                    iter_times_list.append(iter_time)
                    iter_start_time = time.time()
                    print(f'[{phase}] Epoch {epoch}  Iter Time:{iter_time:.3f}  Step:{batch_idx}/{len(dataloaders[phase]):<8}  l: {running_loss/batch_idx:<10}')

            total_loss = running_loss/len(dataloaders[phase])
            epoch_time_str = time.strftime("%H:%M:%S",
                                           time.gmtime(time.time() - epoch_start_time))
            print(f'Totals: {phase}  loss: {total_loss}  time: {epoch_time_str}')
            print()

            if phase == 'train':
                save_checkpoint(epoch=epoch,
                                model=gru,
                                model_params=gru.module.params if multi_gpu else gru.params,
                                optimizer=gru_optimizer,
                                optimizer_params=dict(lr=learning_rate),
                                loss=total_loss,
                                global_step=global_step,
                                save_path=osp.join(save_dir, f'{epoch:02d}.pt'))
            else:
                model_logger.val.add_scalar('loss/iter',
                                            total_loss,
                                            global_step)
                if total_loss < best.loss:
                    best.loss = total_loss
                    best.epoch = epoch
                    if osp.exists(best.path):
                        os.remove(best.path)
                    best.path = osp.join(save_dir, f'best_{epoch:02d}.pt')
                    save_checkpoint(epoch=epoch,
                                    model=gru,
                                    model_params=gru.module.params if multi_gpu else gru.params,
                                    optimizer=gru_optimizer,
                                    optimizer_params=dict(lr=learning_rate),
                                    loss=total_loss,
                                    global_step=global_step,
                                    save_path=best.path)
                else:
                    if (epoch - best.epoch) >= early_stop:
                        print(f'Early Stop @ Epoch {epoch}')
                        print(f'Best model:')
                        print(f'    Epoch:    {best.epoch}')
                        print(f'    Val Loss: {best.loss}')
                        print(f'    Path:     {best.path}')
                        exit(0)

    total_train_time_str = time.strftime("%H:%M:%S",
                                         time.gmtime(sum(iter_times_list)))
    print(f'Total training time: {total_train_time_str}')
    return total_loss


def main(local_config, config):
    # Training params
    start_epoch = 1
    num_epochs = config.training.num_epochs
    batch_size = config.training.batch_sz
    dropout_rate = config.training.dropout
    learning_rate = config.training.learning_rate
    early_stop = config.training.early_stop

    # Data loading params
    num_audio_files = config.data.num_train_files
    dataload_workers = config.data.num_dataload_workers
    data_win_sz = config.data.win_sz
    data_win_stride = config.data.win_stride
    train_dataset_path = osp.join(local_config.datasets_dir,
                                  config.data.train_path)
    valid_dataset_path = osp.join(local_config.datasets_dir,
                                  config.data.valid_path)

    # Model params
    input_size = config.model.input_sz
    hidden_size = config.model.hidden_sz
    output_size = config.model.output_sz
    num_layers = config.model.num_layers
    bidirectional = config.model.bidirectional
    model_save_dir = osp.join(local_config.models_dir, config.model.save_dir)
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    # Log params
    log_dir = osp.join(local_config.logs_dir, config.log.save_dir)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_start_step = 0

    # Create device under which the model will be trained
    cuda_str = 'cpu'
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if config.training.multi_gpu and num_gpus > 1:
            cuda_str = 'cuda'
        else:
            target_gid = int(args.gpu_id)
            if target_gid >= 0 and target_gid < num_gpus:
                cuda_str = f'cuda:{target_gid}'

    print(f'Training on device: {cuda_str}')
    device = torch.device(cuda_str)

    # Create dataset
    if config.data.type == 'mocap':
        print('Loading Training data')
        train_dataset = TongueMocapDataset(train_dataset_path,
                                           num_files=num_audio_files,
                                           win_sz=data_win_sz,
                                           stride=data_win_stride,
                                           pose_only=True)
        print(f'Training samples:    {len(train_dataset)}')

        print('Loading Validation data')
        valid_dataset = TongueMocapDataset(valid_dataset_path,
                                           num_files=num_audio_files,
                                           win_sz=data_win_sz,
                                           stride=data_win_stride,
                                           pose_only=True)
        print(f'Validation samples:  {len(valid_dataset)}')
    else:
        print('ERROR: Unknown dataset type')
        exit(-1)

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=dataload_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=dataload_workers,
                                  pin_memory=True)
    dataloaders = dict(train=train_dataloader,
                       valid=valid_dataloader)

    # Build if no checkpoint is given
    if 'checkpoint' not in config:
        print('Building new model')
        rnn = GRU(input_size, hidden_size, output_size,
                  n_layers=num_layers,
                  dropout=dropout_rate,
                  bidirectional=bidirectional)
        if config.training.multi_gpu:
            rnn = nn.DataParallel(rnn)
        rnn.to(device)

        print('Building new optimizer')
        rnn_optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    else:
        print('Loading model checkpoint')
        checkpoint_path = osp.join(local_config.models_dir,
                                   config.checkpoint.path)
        checkpoint = torch.load(checkpoint_path)
        rnn = GRU(**checkpoint['model_params'])
        rnn.load_state_dict(checkpoint['model_state_dict'])
        rnn.to(device)

        print('Loading optimizer checkpoint')
        rnn_optimizer = optim.Adam(rnn.parameters(),
                                   lr=checkpoint['optimizer_params']['lr'])
        rnn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = int(Path(checkpoint_path).stem) + 1
        log_start_step = checkpoint['global_step']

    print_training_header(learning_rate,
                          dropout_rate,
                          batch_size,
                          rnn,
                          model_save_dir,
                          log_dir)

    train(rnn, rnn_optimizer,
          dataloaders=dataloaders,
          save_dir=model_save_dir,
          log_dir=log_dir,
          batch_size=batch_size,
          learning_rate=learning_rate,
          dropout=dropout_rate,
          start_epoch=start_epoch,
          n_epochs=num_epochs,
          early_stop=early_stop,
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
