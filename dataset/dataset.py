import numpy as np
import torch
import bisect
from torch.utils.data import Dataset
from easydict import EasyDict as edict


class TongueMocapDataset(Dataset):
    '''This class loads the data to train the speech2tongue-anim model
        padding.size (int): how many times the padding value is repeated on both edges
        padding.value (str): "zero" if it is multiple zero tensors
                             "edge" if the value on the edges is repeated per sample
    '''
    def __init__(self, dataset_path, num_files=-1, win_sz=50, stride=1,
                 pose_only=False, padding=dict(size=0, value='zeros')):
        padding = edict(padding)
        # Load the data from the dataset file
        data = np.load(dataset_path, allow_pickle=True).item()

        # Audio features [N x SeqLen x FeatDim]
        self.feat_data = data['X'] if num_files == -1 else data['X'][:num_files]
        # Mocap data [N x SeqLen x 30(3dx10j)]
        self.pos_data = data['Y_pos'] if num_files == -1 else data['Y_pos'][:num_files]
        # Add padding if required
        if padding.size > 0:
            X = self.feat_data
            Y = self.pos_data
            X_padded = list()
            Y_padded = list()
            for x, y in zip(X, Y):
                # Audio feats padding
                if padding.value == 'edge':
                    X_padded.append(np.vstack([np.tile(x[0], (padding.size, 1)), x, np.tile(x[-1], (padding.size, 1))]))
                else:
                    feat_dim = x.shape[1]
                    X_padded.append(np.vstack([np.zeros((padding.size, feat_dim)), x, np.zeros((padding.size, feat_dim))]))
                # Mocap padding
                Y_padded.append(np.vstack([np.tile(y[0], (padding.size, 1)), y, np.tile(y[-1], (padding.size, 1))]))
            self.feat_data = np.array(X_padded)
            self.pos_data = np.array(Y_padded)

        self.rot_data = data['Y_rot'] if num_files==-1 else data['Y_rot'][:num_files]
        self.frames = data['frames'] if num_files==-1 else data['frames'][:num_files]
        self.ids = data['ids'] if num_files==-1 else data['ids'][:num_files]

        # Set the parameters
        self.win_sz = win_sz
        self.stride = stride
        self.pose_only = pose_only

        # Build the upper bound of the number of training sample per audio
        sum_samples = 0
        self.samples_idx = list()
        for num_frames in self.frames:
            sum_samples += len(np.arange(0, num_frames-win_sz, stride))
            self.samples_idx.append(sum_samples)

    def __len__(self):
        '''Returns the total number of samples obtained from window sliding'''
        return self.samples_idx[-1]

    def __getitem__(self, index):
        # Gets the audio file that belongs to that sample index
        data_idx = bisect.bisect(self.samples_idx, index)
        start_pt = 0 if data_idx == 0 else self.samples_idx[data_idx-1]
        start_pos = index - start_pt  # start pos within audio file
        X = torch.tensor(self.feat_data[data_idx][start_pos:start_pos+self.win_sz])
        Y_pos = torch.tensor(self.pos_data[data_idx][start_pos:start_pos+self.win_sz])

        if self.pose_only:
            # audio_feat, pos
            return X, Y_pos

        # audio_feat, pos, rotation
        Y_rot = torch.tensor(self.rot_data[data_idx][start_pos:start_pos+self.win_sz])
        return X, Y_pos, Y_rot

    def get_num_files(self):
        return len(self.ids)

    def get_sample_by_id(self, sample_id):
        sample_idx = self.ids.index(sample_id)
        return self.feat_data[sample_idx], self.pos_data[sample_idx], self.rot_data[sample_idx]

    def get_sample_by_index(self, idx):
        return self.feat_data[idx], self.pos_data[idx], self.rot_data[idx]


class TongueMocapFileDataset(Dataset):
    '''This class loads the data and iterates per full sequence of the data'''
    def __init__(self, dataset_path, num_files=-1, pose_only=False):
        # Load the data from the dataset file
        data = np.load(dataset_path, allow_pickle=True).item()
        self.feat_data = data['X'] if num_files==-1 else data['X'][:num_files]
        self.pos_data = data['Y_pos'] if num_files==-1 else data['Y_pos'][:num_files]
        self.rot_data = data['Y_rot'] if num_files==-1 else data['Y_rot'][:num_files]
        self.frames = data['frames'] if num_files==-1 else data['frames'][:num_files]
        self.ids = data['ids'] if num_files==-1 else data['ids'][:num_files]

        # Meta-params
        self.pose_only = pose_only

    def __len__(self):
        '''Returns the total number of samples obtained from window sliding'''
        return len(self.ids)

    def __getitem__(self, index):
        X = torch.tensor(self.feat_data[index])
        Y_pos = torch.tensor(self.pos_data[index])
        if self.pose_only:
            # audio_feat, pos
            return X, Y_pos

        # audio_feat, pose, rotation
        Y_rot = torch.tensor(self.rot_data[index])
        return X, Y_pos, Y_rot


class CausalTongueMocapDataset(Dataset):
    """Implements a pytorch dataset for a given input window size and
        its corresponding causal output window size

    Args:
        Dataset (Dataset): torch Dataset to be used while training a model
    """

    def __init__(self, dataset_path, num_files=-1,
                 input_win_sz=5, output_win_sz=1, stride=1,
                 pose_only=True):
        # Load the data from the dataset file
        data = np.load(dataset_path, allow_pickle=True).item()
        self.feat_data = data['X'] if num_files == -1 else data['X'][:num_files]
        self.pos_data = data['Y_pos'] if num_files == -1 else data['Y_pos'][:num_files]
        self.rot_data = data['Y_rot'] if num_files == -1 else data['Y_rot'][:num_files]
        self.frames = data['frames'] if num_files == -1 else data['frames'][:num_files]
        self.ids = data['ids'] if num_files == -1 else data['ids'][:num_files]

        # Set the parameters
        self.input_win_sz = input_win_sz
        self.output_win_sz = output_win_sz
        self.stride = stride
        self.pose_only = pose_only

        # Build the upper bound of the number of training sample per audio
        sum_samples = 0
        self.samples_idx = list()
        for num_frames in self.frames:
            sum_samples += len(np.arange(0, num_frames-input_win_sz, stride))
            self.samples_idx.append(sum_samples)

    def __len__(self):
        '''Returns the total number of samples obtained from window sliding'''
        return self.samples_idx[-1]

    def __getitem__(self, index):
        # Gets the audio file that belongs to that sample index
        data_idx = bisect.bisect(self.samples_idx, index)
        start_pt = 0 if data_idx == 0 else self.samples_idx[data_idx-1]
        start_pos = index - start_pt  # start pos within audio file
        end_pos = start_pos + self.input_win_sz
        causal_start_pos = end_pos - self.output_win_sz
        X = torch.tensor(self.feat_data[data_idx][start_pos:end_pos])
        Y_pos = torch.tensor(self.pos_data[data_idx][causal_start_pos:end_pos])
        Y_rot = torch.tensor(self.rot_data[data_idx][causal_start_pos:end_pos])

        if self.pose_only:
            # audio_feat, pos
            return X, Y_pos

        # audio_feat, pose, rotation
        return X, Y_pos, Y_rot

    def get_sample(self, sample_id):
        sample_idx = self.ids.index(sample_id)
        return (self.feat_data[sample_idx],
                self.pos_data[sample_idx],
                self.rot_data[sample_idx])

    def get_num_files(self):
        return len(self.ids)

    def get_sample_by_index(self, idx):
        return self.feat_data[idx], self.pos_data[idx], self.rot_data[idx]

