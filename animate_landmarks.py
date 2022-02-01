from __future__ import absolute_import, division, print_function

import argparse
from pathlib import Path

import numpy as np

from rendertools import MayaviTongueRenderer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str,
                        help='Path to the npy file which has the predicted pose')
    parser.add_argument('-v', '--view', type=str,
                        help='View of the render: left, center, right')
    parser.add_argument('-sz', '--size', type=str, default='630x540',
                        help='Output frame size')
    parser.add_argument('-r', '--rotate', action='store_true',
                        help='Rotate the model over the z-axis during the animation')
    parser.add_argument('-l', '--loop', action='store_true',
                        help='Loop the displayed animation')
    parser.add_argument('-t', '--theme', type=str, default='dark',
                        help='Sets the color theme of the render: dark, light')
    parser.add_argument('-gt', '--ground_truth', action='store_true',
                        help='Indicates if the sample comes from the capture data')
    parser.add_argument('-o', '--output_dir', type=str, default=None,
                        help='Dir where the frames will be stored')
    return parser.parse_args()


def parse_size(size_str):
    size_str = size_str.lower()
    if 'x' not in size_str:
        return None, None

    return tuple(map(int, size_str.split('x')))


def main(opts):
    size = parse_size(opts.size)
    input_path = Path(opts.input_path)
    if opts.ground_truth:
        pose_arr = np.load(input_path, allow_pickle=True).item()['pos']
    else:
        pose_arr = np.load(input_path, allow_pickle=True)
    # Allow single frames to be displayed as well
    if pose_arr.ndim == 1:
        pose_arr = np.expand_dims(pose_arr, axis=0)

    palate_path = None
    palate_path = './viz/res/palate_mesh.obj'
    if not Path(palate_path).exists():
        palate_path = None
    renderer = MayaviTongueRenderer(palate_path, theme=opts.theme)

    view_list = ['left', 'center', 'right'] if opts.view == 'all' else [opts.view]
    for view in view_list:
        save_dir = Path(opts.output_dir, input_path.stem, view)
        save_dir.mkdir(parents=True, exist_ok=True)
        renderer.render_tongue(pose_arr,
                               save_dir=save_dir,
                               anim_delay=10,
                               view=view,
                               size=size,
                               loop=opts.loop,
                               rotate=opts.rotate)


if __name__ == '__main__':
    opts = parse_args()
    main(opts)
