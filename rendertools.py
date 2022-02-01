'''
This renderer is intended to be used to render the Tongue captured data as well
as the output of the models which were trained on the tongue mocap dataset.
Each vector describes a pose and rotation of 10 joints. Each joint has three values
for the pose (X,Y,Z) and another three values for the azimuth, elevation and RMS capturing error.

The order of the joints is the following:
td, tb, br, bl, tt, ul, lc, ll, li, lj
'''

import os.path as osp
import sys

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from easydict import EasyDict as edict
from mayavi import mlab
from tqdm import tqdm

from utils.rotation import azel_matrix


def load_obj(obj_path):
    mesh = o3d.io.read_triangle_mesh(obj_path)
    return np.array(mesh.vertices), np.array(mesh.triangles)


class MayaviTongueRenderer:
    def __init__(self, palate_path=None, theme='dark'):
        # Tongue mesh triangles:
        # td - tb - br
        # td - tb - bl
        # tb - br - td
        # tb - bl - td
        self.CONN = np.array([[0, 1, 2],
                              [0, 1, 3],
                              [1, 2, 4],
                              [1, 3, 4]])

        self.LipsTriangles = np.array([[0, 1, 4],
                                       [0, 2, 6],
                                       [1, 3, 5],
                                       [2, 3, 6]])

        self.palette = dict(pink=(0.87058824, 0.67843137, 0.67843137),
                            pink_gray=(0.81176471, 0.72156863, 0.72156863),
                            mx_pink=(0.90980392, 0.26666667, 0.51764706),
                            magenta=(1., 0., 1.),
                            green=(0.6627451, 0.89019608, 0.53333333),
                            neon_green=(0.666, 1., 0.),
                            mx_green=(0.61176471, 0.77647059, 0.32156863),
                            blue=(0.27058824, 0.53333333, 0.96078431),
                            neon_blue=(0., 0., 1.),
                            cyan=(0.333, 1., 1.),
                            red=(1., 0., 0.),
                            dark_red=(0.333, 0., 0.),
                            orange=(1., 0.666, 0.),
                            yellow=(1., 1., 0.),
                            mauve=(0.66666667, 0.33333333, 0.60784314),
                            purple=(0.33333333, 0., 1.),
                            gray=(.4, .4, .4),
                            light_gray=(.9, .9, .9),
                            white=(1., 1., 1.))

        self.view_angles = dict(lat_left=-90.,
                                left=-42.,
                                center=0.,
                                right=42.,
                                lat_right=90.)

        self.palate = None
        if palate_path is not None:
            palate_vertices, palate_triangles = load_obj(palate_path)
            palate_vertices[:, 2] += 2.0
            self.palate = edict(vertices=palate_vertices,
                                triangles=palate_triangles)

        # Six Upper teeth
        # [[-10.15675328, 18.31255487, -3.04258195],
        #     [-6.65140474, 13.08039633, -3.19505463],
        #     [-4.3145057, 7.8482378, -3.34752732],
        #     [-3.14605619, 2.61607927, -3.5],
        #     [-3.14605619, -2.61607927, -3.5],
        #     [-4.3145057, -7.8482378, -3.34752732],
        #     [-6.65140474, -13.08039633, -3.19505463],
        #     [-10.15675328, -18.31255487, -3.04258195]]

        # Display only two incisors, Goofy style
        self.ui_coords = np.array([0., 0., 0.])
        self.ui_spline = np.array([[-3.14605619, 2.61607927, -3.5],
                                   [-3.14605619, -2.61607927, -3.5]])

        self.theme = self.build_theme(theme)

    def build_theme(self, theme):
        theme_dict = None
        if theme == 'light':
            bg_color = (1., 1., 1.)
            tongue = edict(opacity=0.9)
            palate = edict(opacity=0.2)
            jaw = edict(opacity=0.5,
                        specular=0.0,
                        specular_power=128.)
            lips = edict(line_width=0.1,
                         opacity=0.5,
                         specular=0.0,
                         specular_power=128.)
            landmarks = edict(size=2.2)
            theme_dict = edict(bg_color=bg_color, tongue=tongue,
                               palate=palate, jaw=jaw, lips=lips,
                               landmarks=landmarks)
        else:
            # Dark theme by default
            bg_color = (0., 0., 0.)
            tongue = edict(opacity=0.8)
            palate = edict(opacity=0.5)
            jaw = edict(opacity=0.1,
                        specular=0.0,
                        specular_power=128.)
            lips = edict(line_width=0.1,
                         opacity=0.1,
                         specular=0.0,
                         specular_power=128.)  # specular [0, 128]
            landmarks = edict(size=1.5)
            theme_dict = edict(bg_color=bg_color, tongue=tongue,
                               palate=palate, jaw=jaw, lips=lips,
                               landmarks=landmarks)

        return theme_dict

    def calc_azel_matrix(self, azi, ele):
        azi = np.radians(azi)
        ele = np.radians(ele)

        return azel_matrix(azi, ele)

    def calc_jaw_spline(self, jaw_coords, num_steps=8):
        """ Prototype for only 3 points LI, LJ, LJ' 
            arguments:
                jaw_coords (array): Array with three 3D coords of jaw sensors [3x3]
            returns:
                (array): Array with the list of 3D points for the spline to be displayed"""
        x_jaw = jaw_coords[:, 0]
        y_jaw = jaw_coords[:, 1]
        z_jaw = jaw_coords[:, 2]

        # Parabola is a function of y (coronal plane)
        f = np.poly1d(np.polyfit(y_jaw, x_jaw, 2))

        y_spline = np.linspace(y_jaw[0], y_jaw[2], num_steps)
        x_spline = f(y_spline)

        if num_steps % 2 == 0:
            mid_step = num_steps // 2
            z_spline = np.concatenate([np.linspace(z_jaw[0], z_jaw[1], mid_step),
                                       np.linspace(z_jaw[1], z_jaw[2], mid_step)])
        else:
            mid_step = -(-num_steps // 2)
            z_spline = np.concatenate([np.linspace(z_jaw[0], z_jaw[1], mid_step),
                                       np.linspace(z_jaw[1], z_jaw[2], mid_step)[1:]])

        return np.concatenate([np.transpose(x_spline.reshape(1, -1)),
                               np.transpose(y_spline.reshape(1, -1)), 
                               np.transpose(z_spline.reshape(1, -1))], axis=1)


    def render_poses_traj(self, pos_data, rot_data, save_dir=None, output_size=(640,640), view='left'):
        ''' Renders the tongue pose based on the pose data and outputs jpg frames in the output dir
        params:
            pos_data: np array with the sequence of poses to be rendered
            rot_data: np array with the sequence of azimuth/elevations to be rendered
            output_dir: directory where the frames will be saved
            output_size: size of the frame to be renderered, frames are squared
            view: this can be left, center, right
        '''
        num_frames = len(pos_data)
        view_angle = self.view_angles[view]
        quiver3d_mode = 'arrow'
        quiver3d_scale = 5.
        XYZ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # TODO: remove this hack, based on Mark's suggestion
        rot_data[:, [0, 3, 7, 10, 12]] = 0.

        if save_dir is not None:
            # TODO: verify why first two frames are rendered in a bigger size
            mlab.savefig(filename=osp.join(save_dir, '0000.jpg'), 
                         size=output_size)
            mlab.savefig(filename=osp.join(save_dir, '0000.jpg'), 
                         size=output_size)

        for frame_idx in tqdm(range(num_frames)):
            # Tongue: 5 sensors
            # -- vertices
            tongue_verts = pos_data[frame_idx][:15].reshape(5, 3)
            mlab.triangular_mesh(tongue_verts[:, 0], tongue_verts[:, 1], tongue_verts[:, 2],
                                 self.CONN,
                                 colormap='hsv',
                                 transparent=True,
                                 opacity=1.)
            # -- azimuth/elevation
            tongue_azel = rot_data[frame_idx][:15].reshape(5, 3)[:, :2]
            for v, (az, el) in zip(tongue_verts, tongue_azel):
                R = self.calc_azel_matrix(az, el)
                rot_xyz = np.transpose(np.matmul(R, np.transpose(XYZ)))

                mlab.quiver3d(v[0], v[1], v[2],
                              rot_xyz[0, 0], rot_xyz[0, 1], rot_xyz[0, 2], 
                              color=self.palette['red'],
                              mode=quiver3d_mode,
                              scale_factor=quiver3d_scale)
                mlab.quiver3d(v[0], v[1], v[2], rot_xyz[1, 0],
                              rot_xyz[1, 1], rot_xyz[1, 2],
                              color=self.palette['green'], 
                              mode=quiver3d_mode, scale_factor=quiver3d_scale)
                mlab.quiver3d(v[0], v[1], v[2], rot_xyz[2, 0],
                              rot_xyz[2, 1], rot_xyz[2, 2],
                              color=self.palette['blue'], 
                              mode=quiver3d_mode, scale_factor=quiver3d_scale)
            
            # Lips - 3 sensors
            # -- vertices
            lips_verts = pos_data[frame_idx][15:24].reshape(3, 3)
            mlab.points3d(lips_verts[:, 0], lips_verts[:, 1], lips_verts[:, 2],
                          color=self.palette['pink'],
                          scale_factor=2,
                          transparent=True,
                          opacity=1.)
            # -- tubes
            mlab.plot3d(lips_verts[:, 0], lips_verts[:, 1], lips_verts[:, 2], 
                        color=self.palette['pink'], 
                        tube_radius=0.5,
                        transparent=True,
                        opacity=1.)
            
            # Jaw - 2 sensors
            # -- vertices
            jaw_verts = pos_data[frame_idx][24:30].reshape(2, 3)
            mlab.points3d(jaw_verts[:, 0], jaw_verts[:, 1], jaw_verts[:, 2],
                        color=self.palette['green'],
                        scale_factor=2,
                        transparent=True,
                        opacity=1.)

            # View
            mlab.view(view_angle, 
                      80,
                      174.87838774833392, 
                      [-19.96602563, -5.5082283, -4.12793012])
            
            # Save frame
            if save_dir is not None:
                mlab.savefig(filename=osp.join(save_dir, f'{frame_idx:04d}.jpg'),
                             size=output_size)
            mlab.clf()


    def animate_tongue(self, pose_data, save_dir=None, anim_delay=20, view='right', size=(640,540)):
        fig = mlab.figure(size=size, bgcolor=(1, 1, 1))
        
        tongue_coords = pose_data[:, :15].reshape(-1, 5, 3)
        ul_coords = pose_data[:, 15:18]
        lc_coords = pose_data[:, 18:21]
        ll_coords = pose_data[:, 21:24]
        
        lips_coords = pose_data[:, 15:24].reshape(-1, 3, 3)
        li_coords = pose_data[:, 24:27]
        lj_coords = pose_data[:, 27:30]
        
        lips_delta = lc_coords - ul_coords
        lips_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(lips_delta.shape[0], axis=0), lips_delta)
        lcb_coords = ul_coords + lips_delta
        
        jaw_delta = lj_coords - li_coords
        jaw_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(jaw_delta.shape[0], axis=0), jaw_delta)
        lk_coords = li_coords + jaw_delta
        
        mirror_lips_coords = np.concatenate((lips_coords[:, 0, :].reshape(-1, 1, 3), lcb_coords.reshape(-1, 1, 3), lips_coords[:, 2, :].reshape(-1, 1, 3)), axis=1)

        #Tongue
        tongue_plt = mlab.triangular_mesh(tongue_coords[0, :, 0], tongue_coords[0, :, 1], tongue_coords[0, :, 2],
                                    self.CONN,
                                    colormap='hsv', 
                                    transparent=True,
                                    opacity=1.)
        
        #Lips
        ul_plt = mlab.points3d(ul_coords[0, 0], ul_coords[0, 1], ul_coords[0, 2],
                               color=self.palette['pink'],
                               scale_factor=1.8,
                               transparent=False,
                               opacity=1.)
        lc_plt = mlab.points3d(lc_coords[0, 0], lc_coords[0, 1], lc_coords[0, 2],
                               color=self.palette['pink'],
                               scale_factor=1.,
                               transparent=False,
                               opacity=1.)
        lcb_plt = mlab.points3d(lcb_coords[0, 0], lcb_coords[0, 1], lcb_coords[0, 2],
                                color=self.palette['pink_gray'],
                                scale_factor=1.,
                                transparent=False,
                                opacity=.5)
        ll_plt = mlab.points3d(ll_coords[0, 0], ll_coords[0, 1], ll_coords[0, 2],
                               color=self.palette['pink'],
                               scale_factor=1.8,
                               transparent=False,
                               opacity=1.)
        
        lips_plt = mlab.plot3d(lips_coords[0, :, 0], lips_coords[0, :, 1], lips_coords[0, :, 2], 
                               color=self.palette['pink'],
                               tube_radius=1.,
                               transparent=True,
                               opacity=1.)
        
        mirror_lips_plt = mlab.plot3d(mirror_lips_coords[0, :, 0], mirror_lips_coords[0, :, 1], mirror_lips_coords[0, :, 2], 
                                      color=self.palette['pink_gray'], 
                                      tube_radius=1.,
                                      transparent=True,
                                      opacity=.5)
        
        #Jaw
        li_plt = mlab.points3d(li_coords[0, 0], li_coords[0, 1], li_coords[0, 2],
                               color=self.palette['white'],
                               scale_factor=2.5,
                               transparent=False,
                               mode='cube',
                               opacity=1.)
        lj_plt = mlab.points3d(lj_coords[0, 0], lj_coords[0, 1], lj_coords[0, 2],
                               color=self.palette['white'],
                               scale_factor=2.5,
                               transparent=False,
                               mode='cube',
                               opacity=1.)
        lk_plt = mlab.points3d(lk_coords[0, 0], lk_coords[0, 1], lk_coords[0, 2],
                               color=self.palette['light_gray'],
                               scale_factor=2,
                               transparent=False,
                               mode='cube',
                               opacity=.75)
        
        # View
        view_angle = self.view_angles[view]
        mlab.view(view_angle,
                  83.94471955723412,
                  174.87838774833392,
                  [-19.96602563, -5.5082283, -4.12793012])
        
        @mlab.animate(delay=anim_delay)
        def anim(save_dir=None):
            mlab.gcf()
            saved_frames = False
            while True:
                for t in range(len(ul_coords)):
                    # Tongue
                    tongue_plt.mlab_source.set(x=tongue_coords[t, :, 0], y=tongue_coords[t, :, 1], z=tongue_coords[t, :, 2])
                    # Lips
                    ul_plt.mlab_source.set(x=ul_coords[t, 0], y=ul_coords[t, 1], z=ul_coords[t, 2])
                    lc_plt.mlab_source.set(x=lc_coords[t, 0], y=lc_coords[t, 1], z=lc_coords[t, 2])
                    lcb_plt.mlab_source.set(x=lcb_coords[t, 0], y=lcb_coords[t, 1], z=lcb_coords[t, 2])
                    ll_plt.mlab_source.set(x=ll_coords[t, 0], y=ll_coords[t, 1], z=ll_coords[t, 2])
                    lips_plt.mlab_source.set(x=lips_coords[t, :, 0], y=lips_coords[t, :, 1], z=lips_coords[t, :, 2])
                    mirror_lips_plt.mlab_source.set(x=mirror_lips_coords[t, :, 0], y=mirror_lips_coords[t, :, 1], z=mirror_lips_coords[t, :, 2])
                    # Jaw
                    li_plt.mlab_source.set(x=li_coords[t, 0], y=li_coords[t, 1], z=li_coords[t, 2])
                    lj_plt.mlab_source.set(x=lj_coords[t, 0], y=lj_coords[t, 1], z=lj_coords[t, 2])
                    lk_plt.mlab_source.set(x=lk_coords[t, 0], y=lk_coords[t, 1], z=lk_coords[t, 2])
                    
                    if save_dir is not None and not saved_frames:
                        mlab.savefig(osp.join(save_dir, f'{t:04d}.jpg'), size=size)
                    
                    yield
                if not saved_frames:
                    print(f'Finished saving frames to {save_dir}')
                    print(mlab.view())
                saved_frames = True
                
        anim(save_dir)
        mlab.show()
        mlab.close(all=True)

    def render_tongue_pose(self, pose_data, view='right', size=(640, 540), bg_color=(0, 0, 0)):
        mlab.figure(size=size, bgcolor=bg_color)
        marker_scale_factor = 1.8

        tongue_triags = np.array([[0, 1, 5],
                            [0, 1, 6],
                            [1, 2, 5],
                            [1, 3, 6],
                            [1, 2, 7],
                            [1, 3, 8],
                            [1, 4, 7],
                            [1, 4, 8]])

        lips_triags = np.array([[0, 1, 4],
                                [0, 2, 6],
                                [1, 3, 5],
                                [2, 3, 6],
                                [0, 1, 7],
                                [1, 3, 7],
                                [0, 6, 8],
                                [3, 6, 8]])
        
        # Tongue
        td_coords = pose_data[0:3]
        tb_coords = pose_data[3:6]
        br_coords = pose_data[6:9]
        bl_coords = pose_data[9:12]
        tt_coords = pose_data[12:15]

        brr_coords = 1.30 * br_coords - 0.30 * tb_coords
        bll_coords = 1.30 * bl_coords - 0.30 * tb_coords

        tdr_coords = td_coords.copy()
        tdr_coords[1] = 0.8*br_coords[1] + 0.2*td_coords[1]
        tdl_coords = td_coords.copy()
        tdl_coords[1] = 0.8*bl_coords[1] + 0.2*td_coords[1]

        ttt_coords = 1.15 * tt_coords - 0.15 * tb_coords
        ttr_coords = 1.25 * ((brr_coords + ttt_coords) / 2.) - 0.25 * tb_coords
        ttl_coords = 1.25 * ((bll_coords + ttt_coords) / 2.) - 0.25 * tb_coords
        
        tongue_coords_list = [td_coords, tb_coords, brr_coords, bll_coords, ttt_coords, tdr_coords, tdl_coords, ttr_coords, ttl_coords]
        tongue_coords = np.concatenate(tongue_coords_list).reshape(len(tongue_coords_list), 3)

        # Lips
        ul_coords = pose_data[15:18]
        lc_coords = pose_data[18:21]
        ll_coords = pose_data[21:24]
        # -- Upper lips
        ull_coords = ul_coords + [-1, -3, 1]
        ulr_coords = ul_coords + [-1, 3, 1]
        ulc_coords = ul_coords + [-4, 0, -7.58]
        # -- Lower lips
        llu_coords = ll_coords + [50, 0, 6.]
        llc_coords = llu_coords + [3.0, 0, 0.5]
        # -- Mirrored LC
        lips_delta = lc_coords - ul_coords
        lips_delta = np.multiply(np.array([1., -1., 1.]), lips_delta)
        lcb_coords = ul_coords + lips_delta
        
        lips_coords_list = [lc_coords, ul_coords, ll_coords, lcb_coords, ull_coords, ulr_coords, llu_coords, ulc_coords, llc_coords]
        lips_mesh_coords = np.concatenate(lips_coords_list).reshape(len(lips_coords_list), 3)
        
        # Jaw
        li_coords = pose_data[24:27]
        lj_coords = pose_data[27:30]
        jaw_delta = lj_coords - li_coords
        jaw_delta = np.multiply(np.array([1., -1., 1.]), jaw_delta)
        lk_coords = li_coords + jaw_delta
        jaw_coords = np.vstack((lj_coords, li_coords, lk_coords))
        

        #Tongue
        tongue_mesh = mlab.triangular_mesh(tongue_coords[:, 0], tongue_coords[:, 1], tongue_coords[:, 2],
                                    tongue_triags,
                                    representation='surface',
                                    color=self.palette['red'],
                                    transparent=True,
                                    opacity=0.8)
        tongue_mesh.actor.actor.property.set(edge_visibility=True, line_width=2., render_lines_as_tubes=True, edge_color=self.palette['dark_red'],
                                            shading=True, lighting=True,
                                            specular=0.0, specular_power=128., specular_color=self.palette['red'],
                                            diffuse=1.0)
        
        td_plt = mlab.points3d(td_coords[0], td_coords[1], td_coords[2],
                            color=self.palette['purple'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        tb_plt = mlab.points3d(tb_coords[0], tb_coords[1], tb_coords[2],
                            color=self.palette['magenta'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        br_plt = mlab.points3d(br_coords[0], br_coords[1], br_coords[2],
                            color=self.palette['neon_green'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        bl_plt = mlab.points3d(bl_coords[0], bl_coords[1], bl_coords[2],
                            color=self.palette['orange'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        tt_plt = mlab.points3d(tt_coords[0], tt_coords[1], tt_coords[2],
                            color=self.palette['mauve'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)

        #Lips
        lips_mesh = mlab.triangular_mesh(lips_mesh_coords[:, 0], lips_mesh_coords[:, 1], lips_mesh_coords[:, 2],
                                        lips_triags,
                                        representation='surface',
                                        color=self.palette['pink'],
                                        transparent=False,
                                        opacity=0.9)
        lips_mesh.actor.actor.property.set(edge_visibility=True, line_width=2., render_lines_as_tubes=True, edge_color=self.palette['pink'],
                                            shading=True, lighting=True,
                                            specular=0.0, specular_power=128., specular_color=self.palette['pink'],
                                            diffuse=1.0)
                            
        ul_plt = mlab.points3d(ul_coords[0], ul_coords[1], ul_coords[2],
                            color=self.palette['purple'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        lc_plt = mlab.points3d(lc_coords[0], lc_coords[1], lc_coords[2],
                            color=self.palette['magenta'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        lcb_plt = mlab.points3d(lcb_coords[0], lcb_coords[1], lcb_coords[2],
                            color=self.palette['pink_gray'],
                            scale_factor=1.,
                            transparent=False,
                            opacity=.5)
        ll_plt = mlab.points3d(ll_coords[0], ll_coords[1], ll_coords[2],
                            color=self.palette['mauve'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        
        #Jaw
        jaw_plt = mlab.plot3d(jaw_coords[:, 0], jaw_coords[:, 1], jaw_coords[:, 2], 
                            color=self.palette['light_gray'], 
                            tube_radius=0.25,
                            transparent=True,
                            opacity=1.)
        li_plt = mlab.points3d(li_coords[0], li_coords[1], li_coords[2],
                            color=self.palette['white'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            mode='cube',
                            opacity=1.)
        lj_plt = mlab.points3d(lj_coords[0], lj_coords[1], lj_coords[2],
                            color=self.palette['white'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            mode='cube',
                            opacity=1.)
        lk_plt = mlab.points3d(lk_coords[0], lk_coords[1], lk_coords[2],
                            color=self.palette['light_gray'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            mode='cube',
                            opacity=.75)

        # Palate
        if self.palate is not None:
            palate_mesh = mlab.triangular_mesh(self.palate.vertices[:, 0], self.palate.vertices[:, 1], self.palate.vertices[:, 2],
                                        self.palate.triangles,
                                        representation='surface',
                                        color=self.palette['dark_red'],
                                        transparent=False,
                                        opacity=0.5)
        
        # View
        view_angle = self.view_angles[view]
        mlab.view(view_angle, 
                83.94471955723412, 
                174.87838774833392, 
                [-19.96602563,  -5.5082283 ,  -4.12793012])
        
        mlab.show()
        mlab.close(all=True)

    def render_tongue(self, pose_data, save_dir=None, anim_delay=20, view='right', size=(640,540), loop=False, rotate=False):
        mayavi_fig = mlab.figure(size=size, 
                                bgcolor=tuple(self.theme.bg_color))
        marker_scale_factor = self.theme.landmarks.size

        tongue_triags = np.array([[0, 1, 5],
                                  [0, 1, 6],
                                  [1, 2, 5],
                                  [1, 3, 6],
                                  [1, 2, 7],
                                  [1, 3, 8],
                                  [1, 4, 7],
                                  [1, 4, 8]])

        lips_triags = np.array([[0, 1, 4],
                                [0, 2, 6],
                                [1, 3, 5],
                                [2, 3, 6],
                                [0, 1, 7],
                                [1, 3, 7],
                                [0, 6, 8],
                                [3, 6, 8]])
        
        # Tongue
        td_coords = pose_data[:, 0:3]
        tb_coords = pose_data[:, 3:6]
        br_coords = pose_data[:, 6:9]
        bl_coords = pose_data[:, 9:12]
        tt_coords = pose_data[:, 12:15]

        brr_coords = 1.05 * br_coords - 0.05 * tb_coords
        bll_coords = 1.05 * bl_coords - 0.05 * tb_coords

        tdr_coords = td_coords.copy()
        tdr_coords[:, 1] = 0.8*br_coords[:, 1] + 0.2*td_coords[:, 1]
        tdl_coords = td_coords.copy()
        tdl_coords[:, 1] = 0.8*bl_coords[:, 1] + 0.2*td_coords[:, 1]

        ttt_coords = 1.05 * tt_coords - 0.05 * tb_coords
        ttr_coords = 1.15 * ((brr_coords + ttt_coords) / 2.) - 0.15 * tb_coords
        ttl_coords = 1.15 * ((bll_coords + ttt_coords) / 2.) - 0.15 * tb_coords

        tongue_coords_list = [td_coords, tb_coords, brr_coords, bll_coords, ttt_coords, tdr_coords, tdl_coords, ttr_coords, ttl_coords]
        tongue_coords = np.concatenate(tongue_coords_list, axis=1).reshape(-1, len(tongue_coords_list), 3)
        
        # Lips
        ul_coords = pose_data[:, 15:18]             # Upper Lip
        lc_coords = pose_data[:, 18:21]             # Lip Center (right corner)
        ll_coords = pose_data[:, 21:24]             # Lower Lip
        ull_coords = ul_coords + [-1, -3, 1]        # Upper Lip Left
        ulr_coords = ul_coords + [-1, 3, 1]         # Upper Lip Right
        ulc_coords = ul_coords + [-2, 0, -8.58]     # Upper Lip Center
        llu_coords = ll_coords + [3, 0, 5.5]        # Lower Lip Up
        llc_coords = llu_coords + [-2, 0, 4]        # Lower Lip Center
        lips_delta = lc_coords - ul_coords
        lips_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(lips_delta.shape[0], axis=0), lips_delta)
        lcb_coords = ul_coords + lips_delta
        
        lips_coords_list = [lc_coords, ul_coords, ll_coords, lcb_coords, ull_coords, ulr_coords, llu_coords, ulc_coords, llc_coords]
        lips_mesh_coords = np.concatenate(lips_coords_list, axis=1).reshape(-1, len(lips_coords_list), 3)


        # Jaw
        li_coords = pose_data[:, 24:27]
        lj_coords = pose_data[:, 27:30]
        jaw_delta = lj_coords - li_coords
        jaw_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(jaw_delta.shape[0], axis=0), jaw_delta)
        lk_coords = li_coords + jaw_delta
        jaw_coords = np.concatenate((lj_coords.reshape(-1, 1, 3), li_coords.reshape(-1, 1, 3), lk_coords.reshape(-1, 1, 3)), axis=1)

        #Tongue
        tongue_mesh = mlab.triangular_mesh(tongue_coords[0, :, 0], tongue_coords[0, :, 1], tongue_coords[0, :, 2],
                                           tongue_triags,
                                           representation='surface',
                                           color=self.palette['red'],
                                           transparent=True,
                                           opacity=self.theme.tongue.opacity)
        tongue_mesh.actor.actor.property.set(edge_visibility=True, line_width=2., render_lines_as_tubes=True, edge_color=self.palette['dark_red'],
                                             shading=True, lighting=True,
                                             specular=0.0, specular_power=128., specular_color=self.palette['red'],
                                             diffuse=1.0,)
        td_plt = mlab.points3d(td_coords[0, 0], td_coords[0, 1], td_coords[0, 2],
                               color=self.palette['purple'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        tb_plt = mlab.points3d(tb_coords[0, 0], tb_coords[0, 1], tb_coords[0, 2],
                               color=self.palette['magenta'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        br_plt = mlab.points3d(br_coords[0, 0], br_coords[0, 1], br_coords[0, 2],
                               color=self.palette['neon_green'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        bl_plt = mlab.points3d(bl_coords[0, 0], bl_coords[0, 1], bl_coords[0, 2],
                               color=self.palette['orange'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        tt_plt = mlab.points3d(tt_coords[0, 0], tt_coords[0, 1], tt_coords[0, 2],
                               color=self.palette['mauve'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        
        #Lips
        lips_mesh = mlab.triangular_mesh(lips_mesh_coords[0, :, 0], lips_mesh_coords[0, :, 1], lips_mesh_coords[0, :, 2],
                                         lips_triags,
                                         representation='surface',
                                         color=self.palette['pink'],
                                         transparent=True,
                                         opacity=self.theme.lips.opacity)
        lips_mesh.actor.actor.property.set(edge_visibility=True, line_width=self.theme.lips.line_width, render_lines_as_tubes=True, edge_color=self.palette['pink'],
                                           shading=True, lighting=True,
                                           specular=self.theme.lips.specular, 
                                           specular_power=self.theme.lips.specular_power, 
                                           specular_color=self.palette['white'],
                                           diffuse=1.0)

        ul_plt = mlab.points3d(ul_coords[0, 0], ul_coords[0, 1], ul_coords[0, 2],
                               color=self.palette['purple'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        lc_plt = mlab.points3d(lc_coords[0, 0], lc_coords[0, 1], lc_coords[0, 2],
                               color=self.palette['magenta'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        ll_plt = mlab.points3d(ll_coords[0, 0], ll_coords[0, 1], ll_coords[0, 2],
                               color=self.palette['mauve'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=1.)
        
        #Jaw
        li_plt = mlab.points3d(li_coords[0, 0], li_coords[0, 1], li_coords[0, 2],
                               color=self.palette['neon_blue'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               mode='sphere',
                               opacity=1.)
        lj_plt = mlab.points3d(lj_coords[0, 0], lj_coords[0, 1], lj_coords[0, 2],
                               color=self.palette['cyan'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               mode='sphere',
                               opacity=1.)
        # TODO: parallelize this computation
        jaw_splines = np.stack([self.calc_jaw_spline(jc, num_steps=8) for jc in jaw_coords])[:, 1:-1, :] # display only six teeth
        jaw_teeth_plt = mlab.points3d(jaw_splines[0, :, 0] - 3., jaw_splines[0, :, 1], jaw_splines[0, :, 2] + 3.5, 
                                      color=self.palette['light_gray'],
                                      scale_factor=5.,
                                      transparent=True,
                                      mode='cube',
                                      opacity=self.theme.jaw.opacity)
        jaw_teeth_plt.actor.actor.property.set(edge_visibility=True, line_width=2., render_lines_as_tubes=True, edge_color=self.palette['light_gray'],
                                               shading=True, lighting=True,
                                               specular=self.theme.jaw.specular, 
                                               specular_power=self.theme.jaw.specular_power, 
                                               specular_color=self.palette['light_gray'],
                                               diffuse=1.0)

        # Upper Incissor
        ui_plt = mlab.points3d(self.ui_coords[0], self.ui_coords[1], self.ui_coords[2],
                               color=self.palette['blue'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               mode='sphere',
                               opacity=self.theme.jaw.opacity)

        upper_teeth_plt = mlab.points3d(self.ui_spline[:, 0], self.ui_spline[:, 1], self.ui_spline[:, 2], 
                                        color=self.palette['light_gray'],
                                        scale_factor=5.,
                                        transparent=True,
                                        mode='cube',
                                        opacity=self.theme.jaw.opacity)
        upper_teeth_plt.actor.actor.property.set(edge_visibility=True, line_width=2., render_lines_as_tubes=True, edge_color=self.palette['light_gray'],
                                                 shading=True, lighting=True,
                                                 specular=self.theme.jaw.specular, 
                                                 specular_power=self.theme.jaw.specular_power, 
                                                 specular_color=self.palette['light_gray'],
                                                 diffuse=1.0)

        # Palate
        if self.palate is not None:
            palate_mesh = mlab.triangular_mesh(self.palate.vertices[:, 0], self.palate.vertices[:, 1], self.palate.vertices[:, 2],
                                               self.palate.triangles,
                                               representation='surface',
                                               color=self.palette['dark_red'],
                                               transparent=True,
                                               opacity=self.theme.palate.opacity)

        # View
        view_angle = self.view_angles[view]
        elevation = 95.0 if view not in ['left', 'right'] else 84.7
        mlab.view(view_angle, 
                  elevation,
                  174.87838774833392, 
                  [-19.96602563, -5.5082283, -4.12793012])
        
        
        @mlab.animate(delay=anim_delay)
        def render(save_dir=None):
            # Annoying hack to get the first frame at the correct size
            # TODO: Seek why the first frame is always rendered in a larger size
            mlab.savefig(osp.join(save_dir, f'{0:04d}.jpg'), size=size)
            keep_rendering = True
            saved_frames = False
            step = 360. / len(ul_coords)
            alpha = view_angle
            
            while keep_rendering:
                for t in range(len(ul_coords)):
                    if rotate:
                        alpha = (view_angle + t*step) % 360
                        mlab.view(alpha, 
                                  83.94471955723412, 
                                  174.87838774833392, 
                                  [-19.96602563, -5.5082283, -4.12793012])
                    #Tongue
                    tongue_mesh.mlab_source.set(x=tongue_coords[t, :, 0], y=tongue_coords[t, :, 1], z=tongue_coords[t, :, 2])
                    td_plt.mlab_source.set(x=td_coords[t, 0], y=td_coords[t, 1], z=td_coords[t, 2])
                    tb_plt.mlab_source.set(x=tb_coords[t, 0], y=tb_coords[t, 1], z=tb_coords[t, 2])
                    br_plt.mlab_source.set(x=br_coords[t, 0], y=br_coords[t, 1], z=br_coords[t, 2])
                    bl_plt.mlab_source.set(x=bl_coords[t, 0], y=bl_coords[t, 1], z=bl_coords[t, 2])
                    tt_plt.mlab_source.set(x=tt_coords[t, 0], y=tt_coords[t, 1], z=tt_coords[t, 2])
                    #Lips
                    lips_mesh.mlab_source.set(x=lips_mesh_coords[t, :, 0], y=lips_mesh_coords[t, :, 1], z=lips_mesh_coords[t, :, 2])
                    ul_plt.mlab_source.set(x=ul_coords[t, 0], y=ul_coords[t, 1], z=ul_coords[t, 2])
                    lc_plt.mlab_source.set(x=lc_coords[t, 0], y=lc_coords[t, 1], z=lc_coords[t, 2])
                    ll_plt.mlab_source.set(x=ll_coords[t, 0], y=ll_coords[t, 1], z=ll_coords[t, 2])
                    #Jaw
                    # ---- Cubes with lines ---
                    li_plt.mlab_source.set(x=li_coords[t, 0], y=li_coords[t, 1], z=li_coords[t, 2])
                    lj_plt.mlab_source.set(x=lj_coords[t, 0], y=lj_coords[t, 1], z=lj_coords[t, 2])
                    jaw_teeth_plt.mlab_source.set(x=jaw_splines[t, :, 0] - 3, y=jaw_splines[t, :, 1], z=jaw_splines[t, :, 2] + 3.5)

                    if save_dir is not None and not saved_frames:
                        save_path = osp.join(save_dir, f'{t:04d}.png')
                        mlab.draw()
                        plt.axis('off')
                        mlab.savefig(save_path, size=size)
                    yield

                if not saved_frames:
                    print(f'Finished saving frames to {save_dir}')
                    print(mlab.view())
                    saved_frames = True
                
                if not loop:
                    print(mlab.view())
                    keep_rendering = False
                    mlab.close(all=True)
                    sys.exit(0)

        if len(ul_coords) > 1:
            # Render and save frames of animation
            render(save_dir)
            mlab.show()
        else:
            mlab.show()
        
        print(mlab.view)
        mlab.close(all=True)

    def render_midsagittal(self, pose_data, save_dir=None, anim_delay=20, view='right', size=(640,540), loop=False, rotate=False):
        mlab.figure(size=size, bgcolor=tuple(self.theme.bg_color))
        marker_scale_factor = self.theme.landmarks.size

        # Tongue
        td_coords = pose_data[:, 0:3]
        tb_coords = pose_data[:, 3:6]
        br_coords = pose_data[:, 6:9]
        bl_coords = pose_data[:, 9:12]
        tt_coords = pose_data[:, 12:15]

        tongue_coords_list = [tt_coords, tb_coords, td_coords]
        tongue_coords = np.concatenate(tongue_coords_list, axis=1).reshape(-1, len(tongue_coords_list), 3)
        
        # Lips
        ul_coords = pose_data[:, 15:18]             # Upper Lip
        lc_coords = pose_data[:, 18:21]             # Lip Center (right corner)
        ll_coords = pose_data[:, 21:24]             # Lower Lip

        # Jaw
        li_coords = pose_data[:, 24:27]
        lj_coords = pose_data[:, 27:30]

        #Tongue
        td_plt = mlab.points3d(td_coords[0, 0], td_coords[0, 1], td_coords[0, 2],
                            color=self.palette['purple'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        tb_plt = mlab.points3d(tb_coords[0, 0], tb_coords[0, 1], tb_coords[0, 2],
                            color=self.palette['magenta'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        br_plt = mlab.points3d(br_coords[0, 0], br_coords[0, 1], br_coords[0, 2],
                            color=self.palette['neon_green'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        bl_plt = mlab.points3d(bl_coords[0, 0], bl_coords[0, 1], bl_coords[0, 2],
                            color=self.palette['orange'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        tt_plt = mlab.points3d(tt_coords[0, 0], tt_coords[0, 1], tt_coords[0, 2],
                            color=self.palette['mauve'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        tongue_line = mlab.plot3d(tongue_coords[0, :, 0], tongue_coords[0, :, 1], tongue_coords[0, :, 2], 
                                color=(1., 0., 0.))
        
        #Lips
        ul_plt = mlab.points3d(ul_coords[0, 0], ul_coords[0, 1], ul_coords[0, 2],
                            color=self.palette['purple'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        lc_plt = mlab.points3d(lc_coords[0, 0], lc_coords[0, 1], lc_coords[0, 2],
                            color=self.palette['magenta'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        ll_plt = mlab.points3d(ll_coords[0, 0], ll_coords[0, 1], ll_coords[0, 2],
                            color=self.palette['mauve'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            opacity=1.)
        
        #Jaw
        li_plt = mlab.points3d(li_coords[0, 0], li_coords[0, 1], li_coords[0, 2],
                            color=self.palette['neon_blue'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            mode='sphere',
                            opacity=1.)
        lj_plt = mlab.points3d(lj_coords[0, 0], lj_coords[0, 1], lj_coords[0, 2],
                            color=self.palette['cyan'],
                            scale_factor=marker_scale_factor,
                            transparent=False,
                            mode='sphere',
                            opacity=1.)

        # View
        view_angle = self.view_angles[view]
        elevation = 95.0 if view not in ['left', 'right'] else 84.7
        mlab.view(view_angle, 
                elevation,
                174.87838774833392, 
                [-19.96602563,  -5.5082283 ,  -4.12793012])
        
        
        @mlab.animate(delay=anim_delay)
        def render(save_dir=None):
            # Annoying hack to get the first frame at the correct size
            # TODO: Seek why the first frame is always rendered in a larger size
            mlab.savefig(osp.join(save_dir, f'{0:04d}.jpg'), size=size)
            keep_rendering = True
            saved_frames = False
            step = 360. / len(ul_coords)
            alpha = view_angle
            
            while keep_rendering:
                for t in range(len(ul_coords)):
                    if rotate:
                        alpha = (view_angle + t*step) % 360
                        mlab.view(alpha, 
                                83.94471955723412, 
                                174.87838774833392, 
                                [-19.96602563,  -5.5082283 ,  -4.12793012])
                    #Tongue
                    td_plt.mlab_source.set(x=td_coords[t, 0], y=td_coords[t, 1], z=td_coords[t, 2])
                    tb_plt.mlab_source.set(x=tb_coords[t, 0], y=tb_coords[t, 1], z=tb_coords[t, 2])
                    br_plt.mlab_source.set(x=br_coords[t, 0], y=br_coords[t, 1], z=br_coords[t, 2])
                    bl_plt.mlab_source.set(x=bl_coords[t, 0], y=bl_coords[t, 1], z=bl_coords[t, 2])
                    tt_plt.mlab_source.set(x=tt_coords[t, 0], y=tt_coords[t, 1], z=tt_coords[t, 2])
                    tongue_line.mlab_source.set(x=tongue_coords[t, :, 0], 
                                                y=tongue_coords[t, :, 1], 
                                                z=tongue_coords[t, :, 2])
                    #Lips
                    ul_plt.mlab_source.set(x=ul_coords[t, 0], y=ul_coords[t, 1], z=ul_coords[t, 2])
                    lc_plt.mlab_source.set(x=lc_coords[t, 0], y=lc_coords[t, 1], z=lc_coords[t, 2])
                    ll_plt.mlab_source.set(x=ll_coords[t, 0], y=ll_coords[t, 1], z=ll_coords[t, 2])
                    #Jaw
                    # ---- Cubes with lines ---
                    li_plt.mlab_source.set(x=li_coords[t, 0], y=li_coords[t, 1], z=li_coords[t, 2])
                    lj_plt.mlab_source.set(x=lj_coords[t, 0], y=lj_coords[t, 1], z=lj_coords[t, 2])

                    if save_dir is not None and not saved_frames:
                        save_path = osp.join(save_dir, f'{t:04d}.png')
                        mlab.draw()
                        # imgmap = mlab.screenshot(mode='rgba', antialiased=False)
                        plt.axis('off')
                        mlab.savefig(save_path, size=size)
                        # plt.imsave(arr=imgmap, fname=save_path)
                    yield

                if not saved_frames:
                    print(f'Finished saving frames to {save_dir}')
                    print(mlab.view())
                    saved_frames = True
                
                if not loop:
                    print(mlab.view())
                    keep_rendering = False
                    mlab.close(all=True)
                    sys.exit(0)
        
        if len(ul_coords) > 1:
            # Render and save frames of animation
            render(save_dir)
            mlab.show()
        else:
            mlab.show()
        
        print(mlab.view)
        mlab.close(all=True)

    def structure_pos_data(self, pos_data):
        coords = edict()

        # Tongue
        coords.td = pos_data[:, 0:3]
        coords.tb = pos_data[:, 3:6]
        coords.br = pos_data[:, 6:9]
        coords.bl = pos_data[:, 9:12]
        coords.tt = pos_data[:, 12:15]

        # Lips
        coords.ul = pos_data[:, 15:18]
        coords.lc = pos_data[:, 18:21]
        coords.ll = pos_data[:, 21:24]

        # Jaw
        coords.li = pos_data[:, 24:27]
        coords.lj = pos_data[:, 27:30]

        return coords

    def render_trace(self,
                     gt_data,
                     pose_data,
                     save_dir=None,
                     anim_delay=20,
                     view='right',
                     size=(1080,1080),
                     rotate=False):
        mlab.figure(size=size, bgcolor=tuple(self.theme.bg_color))
        marker_scale_factor = self.theme.landmarks.size

        # --- Mesh triangle definitions
        tongue_triags = np.array([[0, 1, 5],
                            [0, 1, 6],
                            [1, 2, 5],
                            [1, 3, 6],
                            [1, 2, 7],
                            [1, 3, 8],
                            [1, 4, 7],
                            [1, 4, 8]])

        lips_triags = np.array([[0, 1, 4],
                                [0, 2, 6],
                                [1, 3, 5],
                                [2, 3, 6],
                                [0, 1, 7],
                                [1, 3, 7],
                                [0, 6, 8],
                                [3, 6, 8]])
        
        # --- Data points extraction and completion
        gt_coords = self.structure_pos_data(gt_data)
        pred_coords = self.structure_pos_data(pose_data)

        # Tongue

        brr_coords = 1.05 * pred_coords.br - 0.05 * pred_coords.tb
        bll_coords = 1.05 * pred_coords.bl - 0.05 * pred_coords.tb

        tdr_coords = pred_coords.td.copy()
        tdr_coords[:, 1] = 0.8*pred_coords.br[:, 1] + 0.2*pred_coords.td[:, 1]
        tdl_coords = pred_coords.td.copy()
        tdl_coords[:, 1] = 0.8*pred_coords.bl[:, 1] + 0.2*pred_coords.td[:, 1]

        ttt_coords = 1.05 * pred_coords.tt - 0.05 * pred_coords.tb
        ttr_coords = 1.15 * ((brr_coords + ttt_coords) / 2.) - 0.15 * pred_coords.tb
        ttl_coords = 1.15 * ((bll_coords + ttt_coords) / 2.) - 0.15 * pred_coords.tb

        tongue_coords_list = [pred_coords.td, pred_coords.tb, brr_coords, bll_coords, ttt_coords, tdr_coords, tdl_coords, ttr_coords, ttl_coords]
        tongue_coords = np.concatenate(tongue_coords_list, axis=1).reshape(-1, len(tongue_coords_list), 3)
        
        # Lips
        ul_coords = pose_data[:, 15:18]             # Upper Lip
        lc_coords = pose_data[:, 18:21]             # Lip Center (right corner)
        ll_coords = pose_data[:, 21:24]             # Lower Lip
        ull_coords = ul_coords + [-1, -3, 1]        # Upper Lip Left
        ulr_coords = ul_coords + [-1, 3, 1]         # Upper Lip Right
        ulc_coords = ul_coords + [-2, 0, -8.58]     # Upper Lip Center
        llu_coords = ll_coords + [3, 0, 5.5]        # Lower Lip Up
        llc_coords = llu_coords + [-2, 0, 4]        # Lower Lip Center
        lips_delta = lc_coords - ul_coords
        lips_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(lips_delta.shape[0], axis=0), lips_delta)
        lcb_coords = ul_coords + lips_delta
        
        lips_coords_list = [lc_coords, ul_coords, ll_coords, lcb_coords, ull_coords, ulr_coords, llu_coords, ulc_coords, llc_coords]
        lips_mesh_coords = np.concatenate(lips_coords_list, axis=1).reshape(-1, len(lips_coords_list), 3)


        # Jaw
        li_coords = pose_data[:, 24:27]
        lj_coords = pose_data[:, 27:30]
        jaw_delta = lj_coords - li_coords
        jaw_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(jaw_delta.shape[0], axis=0), jaw_delta)
        lk_coords = li_coords + jaw_delta
        jaw_coords = np.concatenate((lj_coords.reshape(-1, 1, 3), li_coords.reshape(-1, 1, 3), lk_coords.reshape(-1, 1, 3)), axis=1)

        # --- Declaring and initializing drawing primitives
        # Tongue meshes
        tongue_opacity = self.theme.tongue.opacity
        
        # Tongue markers
        td_plt = mlab.points3d(pred_coords.td[0, 0], pred_coords.td[0, 1], pred_coords.td[0, 2],
                               color=self.palette['purple'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=tongue_opacity)
        tb_plt = mlab.points3d(pred_coords.tb[0, 0], pred_coords.tb[0, 1], pred_coords.tb[0, 2],
                               color=self.palette['magenta'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=tongue_opacity)
        br_plt = mlab.points3d(pred_coords.br[0, 0], pred_coords.br[0, 1], pred_coords.br[0, 2],
                               color=self.palette['neon_green'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=tongue_opacity)
        bl_plt = mlab.points3d(pred_coords.bl[0, 0], pred_coords.bl[0, 1], pred_coords.bl[0, 2],
                               color=self.palette['orange'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=tongue_opacity)
        tt_plt = mlab.points3d(pred_coords.tt[0, 0], pred_coords.tt[0, 1], pred_coords.tt[0, 2],
                               color=self.palette['mauve'],
                               scale_factor=marker_scale_factor,
                               transparent=False,
                               opacity=tongue_opacity)
        
        # Tongue trace
        def fill_coords(coords, t, trace_len=30):
            """Fills a coordinates array with
               the end and start of the trace"""
            start_t = max(0, t-trace_len)
            filled = np.tile(coords[t], (len(coords), 1))
            filled[:t] = coords[:t]
            if start_t >  0:
                filled[:start_t] = np.tile(coords[start_t], (start_t, 1))
            return filled

        # tongue sensor ids
        tongue_sid = ['td', 'tb', 'br', 'bl', 'tt']
        if view == 'lat_right':
            tongue_sid = ['td', 'tb', 'tt']
        elif view == 'center':
            tongue_sid = ['br', 'bl', 'tt']
        tongue_sid = ['td', 'tb', 'br', 'bl', 'tt']
        

        traces = edict()
        traces.gt, traces.pred = edict(), edict()
        trace_radius = 0.1

        # GT
        trace_color = self.palette['light_gray']
        gt_filled_td = fill_coords(gt_coords.td, 1)
        traces.gt.td = mlab.plot3d(gt_filled_td[:, 0],
                                   gt_filled_td[:, 1],
                                   gt_filled_td[:, 2],
                                   color=trace_color,
                                   tube_radius=trace_radius)
        gt_filled_tb = fill_coords(gt_coords.tb, 1)
        traces.gt.tb = mlab.plot3d(gt_filled_tb[:, 0],
                                gt_filled_tb[:, 1],
                                gt_filled_tb[:, 2],
                                color=trace_color,
                                tube_radius=trace_radius)
        gt_filled_br = fill_coords(gt_coords.br, 1)
        traces.gt.br = mlab.plot3d(gt_filled_br[:, 0],
                                gt_filled_br[:, 1],
                                gt_filled_br[:, 2],
                                color=trace_color,
                                tube_radius=trace_radius)
        gt_filled_bl = fill_coords(gt_coords.bl, 1)
        traces.gt.bl = mlab.plot3d(gt_filled_bl[:, 0],
                                gt_filled_bl[:, 1],
                                gt_filled_bl[:, 2],
                                color=trace_color,
                                tube_radius=trace_radius)
        gt_filled_tt = fill_coords(gt_coords.tt, 1)
        traces.gt.tt = mlab.plot3d(gt_filled_tt[:, 0],
                                gt_filled_tt[:, 1],
                                gt_filled_tt[:, 2],
                                color=trace_color,
                                tube_radius=trace_radius)

        # Prediction
        pred_filled_td = fill_coords(pred_coords.td, 1)
        traces.pred.td = mlab.plot3d(pred_filled_td[:, 0],
                                     pred_filled_td[:, 1],
                                     pred_filled_td[:, 2],
                                     color=self.palette['purple'],
                                     tube_radius=trace_radius)
        pred_filled_tb = fill_coords(pred_coords.tb, 1)
        traces.pred.tb = mlab.plot3d(pred_filled_tb[:, 0],
                                     pred_filled_tb[:, 1],
                                     pred_filled_tb[:, 2],
                                     color=self.palette['magenta'],
                                     tube_radius=trace_radius)
        pred_filled_br = fill_coords(pred_coords.br, 1)
        traces.pred.br = mlab.plot3d(pred_filled_br[:, 0],
                                     pred_filled_br[:, 1],
                                     pred_filled_br[:, 2],
                                     color=self.palette['neon_green'],
                                     tube_radius=trace_radius)
        pred_filled_bl = fill_coords(pred_coords.bl, 1)
        traces.pred.bl = mlab.plot3d(pred_filled_bl[:, 0],
                                     pred_filled_bl[:, 1],
                                     pred_filled_bl[:, 2],
                                     color=self.palette['orange'],
                                     tube_radius=trace_radius)
        pred_filled_tt = fill_coords(pred_coords.tt, 1)
        traces.pred.tt = mlab.plot3d(pred_filled_tt[:, 0],
                                     pred_filled_tt[:, 1],
                                     pred_filled_tt[:, 2],
                                     color=self.palette['mauve'],
                                     tube_radius=trace_radius)

        # View
        view_angle = self.view_angles[view]
        elevation = 95.0 if view not in ['left', 'right'] else 80.7

        if view == 'lat_right':
            mlab.view(78.57675029689695,
                    81.84528478811008,
                    174.878387748332,
                    [-33.91487389,  -5.54095268,  -4.50224917])
        elif view == 'center':
            mlab.view
        
        
        @mlab.animate(delay=anim_delay)
        def render(save_dir=None):
            # Annoying hack to get the first frame at the correct size
            # TODO: Seek why the first frame is always rendered in a larger size
            mlab.savefig(osp.join(save_dir, f'{0:04d}.jpg'), size=size)
            keep_rendering = True
            saved_frames = False
            step = 360. / len(ul_coords)
            alpha = view_angle

            seq_len = min(len(gt_coords.td), len(pred_coords.td))
            
            while keep_rendering:
                for t in range(2, seq_len):
                    if rotate:
                        alpha = (view_angle + t*step) % 360
                        mlab.view(alpha, 
                                83.94471955723412, 
                                174.87838774833392, 
                                [-19.96602563,  -5.5082283 ,  -4.12793012])
                    #Tongue
                    for sid in tongue_sid:
                        pred_filled_sensor = fill_coords(pred_coords[sid], t)
                        traces.pred[sid].mlab_source.set(x=pred_filled_sensor[:, 0],
                                                         y=pred_filled_sensor[:, 1],
                                                         z=pred_filled_sensor[:, 2])
                        
                        gt_filled_sensor = fill_coords(gt_coords[sid], t)
                        traces.gt[sid].mlab_source.set(x=gt_filled_sensor[:, 0],
                                                       y=gt_filled_sensor[:, 1],
                                                       z=gt_filled_sensor[:, 2])

                        if save_dir is not None and not saved_frames:
                            save_path = osp.join(save_dir, f'{t:04d}.png')
                            mlab.draw()
                            plt.axis('off')
                            mlab.savefig(save_path, size=size)

                    yield

                if not saved_frames:
                    print(f'Finished saving frames to {save_dir}')
                    print(mlab.view())
                    saved_frames = True

        if len(ul_coords) > 1:
            # Render and save frames of animation
            render(save_dir)
            mlab.show()
        else:
            mlab.show()

        mlab.show()
        
        print(mlab.view)
        mlab.close(all=True) 
        
    def render_tongue_azel(self, pos_data, rot_data, save_dir=None, anim_delay=20, size=(640,540), view='right'):
        fig = mlab.figure(size=size, bgcolor=(1, 1, 1))
        quiver3d_mode = 'arrow'
        quiver3d_scale = 5.
        XYZ = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        
        # Pose
        # -- tongue
        tongue_coords = pos_data[:, :15].reshape(-1, 5, 3)
        
        # -- lips
        lips_coords = pos_data[:, 15:24].reshape(-1, 3, 3)
        ul_coords = pos_data[:, 15:18]
        lc_coords = pos_data[:, 18:21]
        ll_coords = pos_data[:, 21:24]
        lips_delta = lc_coords - ul_coords
        lips_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(lips_delta.shape[0], axis=0), lips_delta)
        lcb_coords = ul_coords + lips_delta
        
        # -- jaw
        li_coords = pos_data[:, 24:27]
        lj_coords = pos_data[:, 27:30]
        jaw_delta = lj_coords - li_coords
        jaw_delta = np.multiply(np.array([[1., -1., 1.]]).repeat(jaw_delta.shape[0], axis=0), jaw_delta)
        lk_coords = li_coords + jaw_delta
        
        mirror_lips_coords = np.concatenate((lips_coords[:, 0, :].reshape(-1, 1, 3), lcb_coords.reshape(-1, 1, 3), lips_coords[:, 2, :].reshape(-1, 1, 3)), axis=1)
        
        # Azimuth/Elevation
        # TODO: hack to remove the azimuth in mid-sagittal and elevation in para-sagittal
        # rot_data[:, [0, 3, 7, 10, 12]] = 0.
        rot_data[:, [0, 3, 12]] = 0.
        
        tongue_azel = rot_data[:, :15].reshape(-1, 5, 3)[:, :, :2]
        tongue_azel_plt = list()
        for v, (az, el) in zip(tongue_coords[0], tongue_azel[0]):
            R = self.calc_azel_matrix(az, el)
            rot_xyz = np.transpose(np.matmul(R, np.transpose(XYZ)))
            
            joint_axes = list()
            joint_axes.append(mlab.quiver3d(v[0], v[1], v[2], rot_xyz[0, 0], rot_xyz[0, 1], rot_xyz[0, 2], color=self.palette['red'], 
                                mode=quiver3d_mode, scale_factor=quiver3d_scale))
            joint_axes.append(mlab.quiver3d(v[0], v[1], v[2], rot_xyz[1, 0], rot_xyz[1, 1], rot_xyz[1, 2], color=self.palette['green'], 
                                mode=quiver3d_mode, scale_factor=quiver3d_scale))
            joint_axes.append(mlab.quiver3d(v[0], v[1], v[2], rot_xyz[2, 0], rot_xyz[2, 1], rot_xyz[2, 2], color=self.palette['blue'], 
                                mode=quiver3d_mode, scale_factor=quiver3d_scale))
            tongue_azel_plt.append(joint_axes)
        
        
        # 3D Objects
        # -- Tongue
        tongue_plt = mlab.triangular_mesh(tongue_coords[0, :, 0], tongue_coords[0, :, 1], tongue_coords[0, :, 2],
                                    self.CONN,
                                    colormap='hsv', 
                                    transparent=True,
                                    opacity=1.)
        
        # -- Lips
        ul_plt = mlab.points3d(ul_coords[0, 0], ul_coords[0, 1], ul_coords[0, 2],
                            color=self.palette['pink'],
                            scale_factor=1.8,
                            transparent=False,
                            opacity=1.)
        lc_plt = mlab.points3d(lc_coords[0, 0], lc_coords[0, 1], lc_coords[0, 2],
                            color=self.palette['pink'],
                            scale_factor=1.,
                            transparent=False,
                            opacity=1.)
        lcb_plt = mlab.points3d(lcb_coords[0, 0], lcb_coords[0, 1], lcb_coords[0, 2],
                            color=self.palette['pink_gray'],
                            scale_factor=1.,
                            transparent=False,
                            opacity=.5)
        ll_plt = mlab.points3d(ll_coords[0, 0], ll_coords[0, 1], ll_coords[0, 2],
                            color=self.palette['pink'],
                            scale_factor=1.8,
                            transparent=False,
                            opacity=1.)
        
        lips_plt = mlab.plot3d(lips_coords[0, :, 0], lips_coords[0, :, 1], lips_coords[0, :, 2], 
                            color=self.palette['pink'], 
                            tube_radius=1.,
                            transparent=True,
                            opacity=1.)
        
        mirror_lips_plt = mlab.plot3d(mirror_lips_coords[0, :, 0], mirror_lips_coords[0, :, 1], mirror_lips_coords[0, :, 2], 
                            color=self.palette['pink_gray'], 
                            tube_radius=1.,
                            transparent=True,
                            opacity=.5)
        
        # -- Jaw
        li_plt = mlab.points3d(li_coords[0, 0], li_coords[0, 1], li_coords[0, 2],
                            color=self.palette['white'],
                            scale_factor=2.5,
                            transparent=False,
                            mode='cube',
                            opacity=1.)
        lj_plt = mlab.points3d(lj_coords[0, 0], lj_coords[0, 1], lj_coords[0, 2],
                            color=self.palette['white'],
                            scale_factor=2.5,
                            transparent=False,
                            mode='cube',
                            opacity=1.)
        lk_plt = mlab.points3d(lk_coords[0, 0], lk_coords[0, 1], lk_coords[0, 2],
                            color=self.palette['light_gray'],
                            scale_factor=2,
                            transparent=False,
                            mode='cube',
                            opacity=.75)
        
        # MayaVi View
        view_angle = self.view_angles[view]
        mlab.view(view_angle, 
                83.94471955723412, 
                174.87838774833392, 
                [-19.96602563,  -5.5082283 ,  -4.12793012])
        
        # This helps render faster than creating objects in each frame
        @mlab.animate(delay=anim_delay)
        def render(save_dir=None):
            # f = mlab.gcf()
            # Annoying hack to get the first frame at the correct size
            # TODO: Seek why the first frame is always rendered in a larger size
            mlab.savefig(osp.join(save_dir, f'{0:04d}.jpg'), size=size)
            for t in tqdm(range(len(ul_coords))):
                # Tongue
                # -- pose
                tongue_plt.mlab_source.set(x=tongue_coords[t, :, 0], y=tongue_coords[t, :, 1], z=tongue_coords[t, :, 2])
                # -- rotation
                for idx, (v, azel) in enumerate(zip(tongue_coords[t], tongue_azel[t])):
                    az, el = azel
                    R = self.calc_azel_matrix(az, el)
                    rot_xyz = np.transpose(np.matmul(R, np.transpose(XYZ)))
                    tongue_azel_plt[idx][0].mlab_source.set(x=v[0], y=v[1], z=v[2], u=rot_xyz[0, 0], v=rot_xyz[0, 1], w=rot_xyz[0, 2])
                    tongue_azel_plt[idx][1].mlab_source.set(x=v[0], y=v[1], z=v[2], u=rot_xyz[1, 0], v=rot_xyz[1, 1], w=rot_xyz[1, 2])
                    tongue_azel_plt[idx][2].mlab_source.set(x=v[0], y=v[1], z=v[2], u=rot_xyz[2, 0], v=rot_xyz[2, 1], w=rot_xyz[2, 2])
                # Lips
                ul_plt.mlab_source.set(x=ul_coords[t, 0], y=ul_coords[t, 1], z=ul_coords[t, 2])
                lc_plt.mlab_source.set(x=lc_coords[t, 0], y=lc_coords[t, 1], z=lc_coords[t, 2])
                lcb_plt.mlab_source.set(x=lcb_coords[t, 0], y=lcb_coords[t, 1], z=lcb_coords[t, 2])
                ll_plt.mlab_source.set(x=ll_coords[t, 0], y=ll_coords[t, 1], z=ll_coords[t, 2])
                lips_plt.mlab_source.set(x=lips_coords[t, :, 0], y=lips_coords[t, :, 1], z=lips_coords[t, :, 2])
                mirror_lips_plt.mlab_source.set(x=mirror_lips_coords[t, :, 0], y=mirror_lips_coords[t, :, 1], z=mirror_lips_coords[t, :, 2])
                # Jaw
                li_plt.mlab_source.set(x=li_coords[t, 0], y=li_coords[t, 1], z=li_coords[t, 2])
                lj_plt.mlab_source.set(x=lj_coords[t, 0], y=lj_coords[t, 1], z=lj_coords[t, 2])
                lk_plt.mlab_source.set(x=lk_coords[t, 0], y=lk_coords[t, 1], z=lk_coords[t, 2])
                
                if save_dir is not None:
                    mlab.draw()
                    mlab.savefig(osp.join(save_dir, f'{t:04d}.jpg'), size=size)
                yield
                
            print(f'Finished saving frames to {save_dir}')
            print(mlab.view())
            mlab.close(all=True)
            sys.exit(0)
        
        render(save_dir)
        mlab.show()
        mlab.close(all=True)
