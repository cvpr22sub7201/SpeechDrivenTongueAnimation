# -*- coding: utf-8 -*-
"""
IO Utils
"""

__version__ = "0.2.0"
__all__ = [
    'get_checkpoint',
    'ensure_dir',
    'get_filename_from_path',
    'file_exists',
    'get_format',
    'write_json',
    'read_from_json',
    'abs_path',
    'get_dir',
    'write_h5',
    'read_h5',
    'get_sub_dirs',
    'get_files',
    'remove_files',
    'make_relative',
]

import re
import os
import json
import h5py
import numpy as np


def get_checkpoint(resume_path: str) -> str:
    """Get checkpoint file from dir with multiple checkpoints

    Args:
        resume_path (str): path to the dir containing the checkpoint or
                           partially defined path ('*' only allowed at the end)

    Raises:
        IOError: No checkpoint has been found

    Returns:
        str: path to the checkpoint file
    """

    if not os.path.isdir(resume_path):

        assert '.pth.tar' not in resume_path

        # path needs to be completed
        partial_name = resume_path.split('/')[-1]
        path_dir = resume_path.replace(partial_name, '')

        models_list = [f for f in os.listdir(path_dir) if partial_name in f]

        if models_list:
            return os.path.join(path_dir, models_list[-1])

        # let's get a model in the same directory
        resume_path = path_dir

    models_list = [f for f in os.listdir(
        resume_path) if f.endswith(".pth.tar")]
    models_list.sort()

    if not models_list:
        raise IOError(
            'Directory {} does not contain any model'.format(resume_path))

    model_name = models_list[-2]

    return os.path.join(resume_path, model_name)


def ensure_dir(path: str) -> str:
    """Make sure directory exists, otherwise create it.

    Args:
        path (str): path to the directory.

    Returns:
        str: path to the directory.
    """

    dir_path = get_dir(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return path


def get_filename_from_path(path: str) -> str:
    """Get name of a file given the absolute or
    relative path.

    Args:
        path (str): path to the file.

    Returns:
        str: file name without format.
    """

    assert isinstance(path, str)
    split = os.path.split(path)
    if len(split) == 1:
        return split

    return split[-1]


def file_exists(path: str) -> bool:
    """Check if file exists

    Args:
        path (str): path to the file.

    Returns:
        bool: True if file exists.
    """

    assert isinstance(path, str)
    return os.path.exists(path)


def write_json(path: str, data: dict) -> None:
    """Save data into a json file

    Args:
        path (str): path where to save the file
        data (dict): data to be stored
    """

    assert isinstance(path, str)
    with open(path, 'w') as out_file:
        json.dump(data, out_file, indent=2)


def read_from_json(path: str) -> dict:
    """Read data from json file

    Args:
        path (str): path to json file

    Raises:
        IOError: File not found

    Returns:
        dict: dictionary containing data
    """

    assert isinstance(path, str)
    if '.json' not in path:
        raise IOError('Path does not point to a json file')

    with open(path, 'r') as in_file:
        data = json.load(in_file)

    return data


def abs_path(path: str) -> str:
    """Get absolute path of a relative one

    Args:
        path (str): relative path

    Raises:
        NameError: string is empty

    Returns:
        str: absolute path
    """

    assert isinstance(path, str)
    if path:
        return os.path.expanduser(path)

    raise NameError('Path is empty...')


def get_dir(path: str) -> str:
    """Get file path from absolute or relative path without
    file name.

    Args:
        path (str): path to directory.

    Returns:
        str: path.
    """

    assert isinstance(path, str), "Path must be a string"

    # check if exists
    if not os.path.exists(path):
        if '.' not in path[:-5]:
            return path
    else:
        # check if it's arready a dir
        if os.path.isdir(path):
            return path

    split = os.path.split(path)
    # check if the path is only the file name
    if len(split) == 1:
        return './'

    return split[0]


def get_format(path: str) -> str:
    """Get file format from path

    Args:
        path (str): file path.

    Raises:
        RuntimeError: File path has no format.

    Returns:
        str: file format.
    """

    file_name = get_filename_from_path(path)
    split_str = file_name.split('.')

    if len(split_str) <= 1:
        raise RuntimeError('File name has no format')

    return split_str[-1]


def write_h5(path: str, data: dict) -> None:
    """Write h5 file

    Args:
        path (str): file path where to save the data
        data (dict): data to be saved

    Raises:
        NotImplementedError: non serializable data to save
    """

    if '.h5' not in path[-3:]:
        path += '.h5'

    hf = h5py.File(path, mode='w')

    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v[0], str):
                v = [a.encode('utf8') for a in v]
            hf.create_dataset(k, data=v)
    elif isinstance(data, list):
        hf.create_dataset('val', data=data)
    elif isinstance(data, np.ndarray):
        hf.create_dataset('val', data=data)
    else:
        raise NotImplementedError

    hf.close()


def read_h5(path: str) -> dict:
    """Load data from h5 file

    Args:
        path (str): file path

    Raises:
        FileNotFoundError: path not pointing to a file

    Returns:
        dict: dictionary containing the data
    """

    data_files = dict()
    h5_data = h5py.File(path, mode='r')

    tags = list(h5_data.keys())
    for tag in tags:
        tag_data = np.asarray(h5_data[tag]).copy()
        data_files.update({tag: tag_data})
    h5_data.close()

    return data_files


def get_sub_dirs(path: str):
    """Get sub-directories contained in a specified directory

    Args:
        path (str): path to directory

    Returns:
        str: lists of absolute paths
        str: list of dir names
    """

    try:
        dirs = os.walk(path).next()[1]
    except AttributeError:
        try:
            dirs = next(os.walk(path))[1]
        except StopIteration:
            dirs = []

    dirs.sort()
    dir_paths = [os.path.abspath(os.path.join(path, dir)) for dir in dirs]

    return dirs, dir_paths


def get_files(path: str, file_format: str, keep_format: bool = True):
    """Get file paths of files contained in
    a given directory according to the format

    Args:
        path (str): path to the directory containing the files
        file_format (list): list or single format
        keep_format (bool, optional): keep format in file names.
                                      Defualts to True.

    Returns:
        str: lists of absolute paths
        str: list of file names
    """

    if isinstance(file_format, str):
        file_format = [file_format]
    else:
        assert isinstance(file_format, list)

    # get sub-directories files
    _, sub_dirs = get_sub_dirs(path)
    if sub_dirs:
        list_paths = []
        list_names = []
        for sub_dir in sub_dirs:
            paths, names = get_files(sub_dir, file_format)
            list_paths.extend(paths)
            list_names.extend(names)
        return list_paths, list_names

    # get current files
    file_names = []
    file_paths = []
    for f_format in file_format:
        if f_format != '*':
            files = [f for f in os.listdir(path)
                     if re.match(r'.*\.{}'.format(f_format), f)]
        else:
            files = os.listdir(path)
        files.sort()

        if not keep_format:
            file_names.extend([f.replace('.{}'.format(f_format), '')
                               for f in files])
        else:
            file_names.extend(files)
        file_paths.extend([os.path.join(path, f) for f in files])

    return file_paths, file_names


def remove_files(paths: list) -> None:
    """Delete files

    Args:
        paths (list): list of paths
    """

    assert isinstance(paths, list)
    for path in paths:
        os.remove(path)


def make_relative(path: str, root_path: str) -> str:
    """Make path relative with respect to a
    root directory

    Arguments:
        path (str): current path.
        root_path (str): root directory path.

    Returns:
        str: relative path.
    """

    r_path = path.replace(root_path, '')
    if r_path:
        if r_path[0] == '/':
            r_path = r_path[1:]

    return r_path
