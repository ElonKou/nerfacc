# !/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================*/
#  Copyright (C)2023 All rights reserved.
#  FileName : main.py
#  Author   : dlkou
#  Email    : elonkou@ktime.cc
#  Date     : Fri 17 Mar 2023 05:33:14 PM CST
# ================================================================*/

import taichi as ti
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import torch

from train_ngp import *

ti.init(arch=ti.cuda, device_memory_GB=1)


class OrbitCamera:
    def __init__(self, K, img_wh, poses, r):
        self.K = K
        self.W, self.H = img_wh
        self.radius = r  # distance of camera.
        self.center = np.zeros(3)  # position of camera.

        pose_np = poses.cpu().numpy()
        # choose a pose as the initial rotation
        self.rot = pose_np[0][:3, :3]

        self.rotate_speed = 0.8
        self.res_defalut = pose_np[0]

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def reset(self, pose=None):
        self.rot = np.eye(3)
        self.center = np.zeros(3)
        self.radius = 4.5
        if pose is not None:
            self.rot = pose.cpu().numpy()[:3, :3]

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(100 * self.rotate_speed * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-100 * self.rotate_speed * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ R.from_rotvec(rotvec_x).as_matrix() @ self.rot

    def scale(self, delta):
        self.radius *= 1.3**(-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-3 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, K, img_wh, poses, radius, model):
        self.model = model

        self.poses = poses

        self.cam = OrbitCamera(K, img_wh, poses, r=radius)
        self.W, self.H = img_wh
        self.render_buffer = ti.Vector.field(3, dtype=float, shape=(self.W, self.H))

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

    def render_cam(self):
        t = time.time()
        tot = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        tot = np.array([np.matmul(tot, self.cam.pose)])
        c2w = torch.Tensor(tot).to("cuda:0")
        self.rgb, self.acc, self.depth = self.model.predict(c2w)
        self.dt = time.time() - t

    def check_cam_rotate(self, window, last_orbit_x, last_orbit_y):
        if window.is_pressed(ti.ui.RMB):
            curr_mouse_x, curr_mouse_y = window.get_cursor_pos()
            if last_orbit_x is None or last_orbit_y is None:
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
            else:
                dx = curr_mouse_y - last_orbit_y
                dy = curr_mouse_x - last_orbit_x
                self.cam.orbit(dx, dy)
                last_orbit_x, last_orbit_y = curr_mouse_x, curr_mouse_y
        else:
            last_orbit_x = None
            last_orbit_y = None

        return last_orbit_x, last_orbit_y

    def check_key_press(self, window):
        if window.is_pressed('w'):
            self.cam.scale(0.2)
        if window.is_pressed('s'):
            self.cam.scale(-0.2)
        if window.is_pressed('a'):
            self.cam.pan(0., 100.)
        if window.is_pressed('d'):
            self.cam.pan(0., -100.)
        if window.is_pressed('e'):
            self.cam.pan(-100, 0.)
        if window.is_pressed('q'):
            self.cam.pan(100, 0.)
        if window.is_pressed(ti.ui.ESCAPE):
            window.running = False
        # print(self.cam.pose)

    def render(self):
        window = ti.ui.Window('NGP demo', (self.W, self.H),)
        canvas = window.get_canvas()
        gui = window.get_gui()

        # GUI controls variables
        last_orbit_x = None
        last_orbit_y = None

        view_id = 0
        last_view_id = 0

        views_size = self.poses.shape[0] - 1

        # render
        self.old_cam_pos = None

        while window.running:
            self.check_key_press(window)
            last_orbit_x, last_orbit_y = self.check_cam_rotate(window, last_orbit_x, last_orbit_y)
            with gui.sub_window("Control", 0.01, 0.01, 0.4, 0.2) as w:
                self.cam.rotate_speed = w.slider_float('rotate speed', self.cam.rotate_speed, 0.1, 2.)
                self.img_mode = w.slider_int("show mode", self.img_mode, 0, 2)
                view_id = w.slider_int('train view', view_id, 0, views_size)

                if last_view_id != view_id:
                    last_view_id = view_id
                    self.cam.reset(self.poses[view_id])

                w.text(f'samples per rays: {self.mean_samples:.2f} s/r')
                w.text(f'render times: {1000*self.dt:.2f} ms')

            if self.old_cam_pos is None or not (self.old_cam_pos == self.cam.pose).all():
                self.render_cam()
                self.old_cam_pos = self.cam.pose.copy()

            if self.img_mode == 0:
                self.render_buffer.from_torch(self.rgb)
                canvas.set_image(self.render_buffer)
            elif self.img_mode == 1:
                self.render_buffer.from_torch(self.depth / 10.0)
                canvas.set_image(self.render_buffer)
            else:
                self.render_buffer.from_torch(self.acc)
                canvas.set_image(self.render_buffer)

            window.show()


if __name__ == "__main__":
    args.run_type = "predict"
    args.scene = "ek"
    args.img_h = 800
    args.img_w = 800
    kwargs = {
        'root_dir': "/mnt/data/DATASET/NERF_dataset/NERF_my/",
        'downsample': 1.0,
    }
    img_wh = (800, 800)
    K = torch.Tensor([[1.1111e+03, 0.0000e+00, 4.0000e+02],
                      [0.0000e+00, 1.1111e+03, 4.0000e+02],
                      [0.0000e+00, 0.0000e+00, 1.0000e+00]])
    poses = torch.Tensor([[[-9.9990e-01, -4.1922e-03,  1.3346e-02, -7.6729e-03],
                           [-1.3989e-02,  2.9966e-01, -9.5394e-01,  1.4358e+00],
                           [-4.6566e-10, -9.5404e-01, -2.9969e-01,  3.7540e-01]],

                          [[-9.3054e-01, -1.1708e-01,  3.4696e-01, -4.9283e-01],
                           [-3.6618e-01,  2.9751e-01, -8.8170e-01,  1.3307e+00],
                           [7.4506e-09, -9.4751e-01, -3.1972e-01,  4.0453e-01]]]).to("cuda:0")
    model = NerfNGP()
    model.load_model()
    NGPGUI(K, img_wh, poses, 4.5, model).render()
