import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import tqdm
import yaml as pyaml
from easydict import EasyDict as edict
from ruamel.yaml import YAML
from tqdm import tqdm

from models import MLP, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--local_config_path', type=str,
                        help='Path to the local machine configuration')
    parser.add_argument('-c', '--config_path', type=str,
                        help='Path to the experiment configuration')
    parser.add_argument('-f', '--feature_type', type=str,
                        help='Type of feature to be predicted on: ds, mfcc, w2vc, w2vz, all')
    parser.add_argument('-m', '--model_type', type=str, default='mlp',
                        help='Type of model to be used')
    parser.add_argument('-gid', '--gpu_id', type=int, default=0,
                        help='GPU to be used for training')
    return parser.parse_args()


def pre_inv_diag_indices(n):
    x = list(range(n))
    y = list(range(n))[::-1]
    return list(zip(x, y))


def post_inv_diag_indices(r, j, n):
    x = [r-c for c in range(n - j)]
    y = list(range(j, n))
    return list(zip(x, y))


def calc_mean_sequence(pose_pred, out_win_sz):
    num_rows = len(pose_pred)
    num_cols = out_win_sz
    means = list()
    for i in range(0, num_rows):
        # Before it reaches the column width
        if i < num_cols-1:
            acc = list()
            for x, y in pre_inv_diag_indices(i + 1):
                acc.append(pose_pred[x][y])
            means.append(torch.mean(torch.stack(acc), dim=0))
        else:
            # The row is equal or larger than column width
            acc = list()
            for x, y in pre_inv_diag_indices(num_cols):    
                x += i - (num_cols-1)
                acc.append(pose_pred[x][y])
            means.append(torch.mean(torch.stack(acc), dim=0))
            

    # Append the lower diagonal
    for j in range(1, num_cols):
        acc = list()
        for x, y in post_inv_diag_indices(num_rows-1, j, num_cols):
            acc.append(pose_pred[x][y])
        means.append(torch.mean(torch.stack(acc), dim=0))
        
    return torch.stack(means)


def export_to_json(pose_arr, channel_labels, model_desc, model_path, fps, proc_time, feature_type, input_file, input_type):
    output = dict()
    output['sequence_length'] = len(pose_arr)
    output['model_desc'] = model_desc
    output['model_path'] = model_path
    output['FPS'] = fps
    output['process_time'] = proc_time
    output['feature_type'] = feature_type
    output['input_file'] = input_file
    output['input_type'] = input_type
    output['sequence'] = dict()
    
    for step, pose in enumerate(pose_arr):
        step_dict = dict()
        for idx, sensor_label in zip(range(0, len(channel_labels)*3, 3), channel_labels):
            step_dict[sensor_label] = dict(pose=pose_arr[step, idx:idx+3].tolist())
        output['sequence'][step] = step_dict 
    
    return json.dumps(output, indent=4)


def main(local_config, config, args):
    # args.feature_type, args.model_type, args.gpu_id
    # Load device
    device_str = 'cpu'
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if args.gpu_id >= 0 and args.gpu_id < num_gpus:
            device_str = f'cuda:{args.gpu_id}'
    device = torch.device(device_str)
    print(f'Device set to {device_str}')
        
    print('Setting model type')
    model_class = MLP

    # Load the checkpoint metadata
    checkpoint_path = Path(local_config.models_dir, config.models[args.model_type][args.feature_type].path)
    checkpoint_name = config.models[args.model_type][args.feature_type].name

    in_win_sz = config.models[args.model_type][args.feature_type].in_win_sz
    out_win_sz = config.models[args.model_type][args.feature_type].out_win_sz
    padding_sz = in_win_sz - out_win_sz
    stride = config.models[args.model_type][args.feature_type].stride

    # Load the checkpoint 
    mlp = load_model(checkpoint_path, model_class, device)
    mlp.eval().to(device)
    
    features_dir = Path(local_config.animation_dir, config.features_dirs[args.feature_type])

    for sample_feats_path in tqdm(features_dir.glob('*.npy')):
        # Load the audio features
        sample_feats = np.load(sample_feats_path, allow_pickle=True)
        padding_tensor = np.tile(sample_feats[0], (padding_sz, 1))
        sample_feats = np.concatenate((padding_tensor, sample_feats))
    
        with torch.no_grad():
            # Estimate the pose
            pose_pred = list()
            for start_idx in range((len(sample_feats) - (in_win_sz-1))):
                x = torch.Tensor(sample_feats[start_idx:start_idx+in_win_sz]) # Network requires a batch
                x = torch.flatten(x).unsqueeze(0).to(device).float()
                y = mlp(x)
                if out_win_sz > 1:
                    pose_pred.append(y.detach().squeeze(0).reshape(out_win_sz, -1))
                else:
                    pose_pred.append(y.detach().squeeze(0)) 
            
            if out_win_sz > 1:
                pose_pred = calc_mean_sequence(pose_pred, out_win_sz)
            else:
                pose_pred = torch.stack(pose_pred, dim=0)

            pose_pred = pose_pred.detach().cpu().numpy()

            # Save the results to NPY
            save_dir = Path(local_config.animation_dir, config.output_dirs[args.feature_type])
            npy_save_path = Path(save_dir, checkpoint_name, 'npy', sample_feats_path.name)
            npy_save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_save_path, pose_pred)

            # Save the results to JSON
            # Export header
            channel_labels = ['td', 'tb', 'br', 'bl', 'tt', 'ul', 'lc', 'll', 'li', 'lj']
            json_str = export_to_json(pose_pred, channel_labels,
                                        model_desc=str(mlp).replace('\n', ''),
                                        model_path=str(config.models[args.model_type][args.feature_type].path),
                                        fps=50.0,
                                        proc_time=datetime.now().strftime('%d/%m/%Y %H:%M:%S'), 
                                        feature_type=args.feature_type,
                                        input_file=f'{sample_feats_path.stem}.wav',
                                        input_type='audio')
            json_save_path = Path(save_dir, checkpoint_name, 'json', f'{sample_feats_path.stem}.json')
            json_save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_save_path, 'w') as output_file:
                output_file.write(json_str)


if __name__ == '__main__':
    # Parse input arguments
    args = parse_args()
    # Load configurations
    yaml = YAML(typ='safe')
    # -- machine configuration
    local_config = edict(yaml.load(open(args.local_config_path)))
    # -- training configuration
    config = edict(yaml.load(open(args.config_path)))
    
    main(local_config, config, args)
