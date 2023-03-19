# !/bin/env python3
# -*- coding: utf-8 -*-
# ================================================================*/
#  Copyright (C)2023 All rights reserved.
#  FileName : train_ngp.py
#  Author   : dlkou
#  Email    : elonkou@ktime.cc
#  Date     : Fri 17 Mar 2023 04:15:59 PM CST
# ================================================================*/


import argparse
import time

import os
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from yaml import parse
from zmq import device
from radiance_fields.ngp import NGPRadianceField
from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    enlarge_aabb,
    render_image,
    set_random_seed,
)

from nerfacc import OccupancyGrid
from datasets.nerf_360_v2 import SubjectLoader
from datasets.nerf_synthetic import SubjectLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/mnt/data/DATASET/NERF_dataset/NERF_my/", help="the root dir of the dataset")
parser.add_argument("--train_split", type=str, default="train", choices=["train", "trainval"], help="which train split to use")
parser.add_argument("--scene", type=str, default="cloth", choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES, help="which scene to use")
parser.add_argument("--test_chunk_size", type=int, default=8192)

parser.add_argument("--max_steps", type=int, default=2000, help="max steps for train.")
parser.add_argument("--sub_steps", type=int, default=200, help="how many steps for save check.")
parser.add_argument("--run_type", type=str, default="train", choices=["train", "val", "predict"], help="train data or validate data.")
parser.add_argument("--save", action="store_true", default=True, help="whether save best model.")
parser.add_argument("--save_image", action="store_true", default=True, help="whether save test image model.")
parser.add_argument("--img_w", type=int, default=800, help="image width and height.")
parser.add_argument("--img_h", type=int, default=800, help="image width and height.")
args = parser.parse_args()


class ExpHelper:
    def __init__(self, is_train=False, parent_exp="cloth") -> None:
        self.is_train = is_train
        self.exp_root = "./exps"
        self.parent_exp = parent_exp
        self.exp_path = ""
        self.exp_name = ""  # cueent exp name
        self.get_exp_name()
        self.rgb_path = os.path.join(self.exp_path, "rgb")  # current rgb path
        self.error_path = os.path.join(self.exp_path, "error")  # current error path
        self.depth_path = os.path.join(self.exp_path, "depth")  # current depth path
        if is_train:
            self.check_create(self.rgb_path)
            self.check_create(self.error_path)
            self.check_create(self.depth_path)

    def check_create(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_all_folders(self, folder):
        folders = []
        for name in os.listdir(folder):
            folder_path = os.path.join(folder, name)
            if os.path.isdir(folder_path):
                folders.append(folder_path)
        return folders

    def get_exp_name(self):
        index_id = 0
        all_folder = os.path.join(self.exp_root, self.parent_exp)  # current exp path = os.path.join(self.exp_root, exp_name)  # the whole folder of exp
        self.check_create(all_folder)
        folders = sorted(self.get_all_folders(all_folder))
        if len(folders) > 0:
            index_id = int(folders[-1][-2:])
        if self.is_train:
            index_id = index_id + 1
            self.exp_name = self.parent_exp + "_" + ("00" + str(index_id))[-2:]  # sub fodler name
            self.exp_path = os.path.join(all_folder, self.exp_name)
        else:
            if index_id > 0:
                self.exp_path = folders[-1]
                self.exp_name = self.exp_path.split("/")[-1]
            else:
                print("no exp folder, please use train.")


class NerfNGP:
    def __init__(self) -> None:
        self.device = "cuda:0"
        set_random_seed(42)
        self.test_dataset = None
        self.train_dataset = None
        self.get_config()
        self.setup()
        self.exp_helper = ExpHelper(args.run_type == "train", args.scene)

    def setup(self):
        # load dataset.
        if args.run_type == "test" or args.run_type == "predict":
            self.test_dataset = SubjectLoader(subject_id=args.scene, root_fp=args.data_root, split="test", num_rays=None, device=self.device, **self.config["test_dataset_kwargs"], WIDTH=args.img_w, HEIGHT=args.img_h)
        elif args.run_type == "train":
            self.train_dataset = SubjectLoader(subject_id=args.scene, root_fp=args.data_root, split=args.train_split, num_rays=self.config["init_batch_size"], device=self.device, **self.config["train_dataset_kwargs"], WIDTH=args.img_w, HEIGHT=args.img_h)

        # setup scene aabb.
        self.scene_aabb = enlarge_aabb(self.config["aabb"], 1 << (self.config["grid_nlvl"] - 1))

        # setup the radiance field we want to train.
        self.radiance_field = NGPRadianceField(aabb=self.scene_aabb).to(self.device)
        self.occupancy_grid = OccupancyGrid(roi_aabb=self.config["aabb"], resolution=self.config["grid_resolution"], levels=self.config["grid_nlvl"]).to(self.device)

        if args.run_type == "train":
            self.grad_scaler = torch.cuda.amp.GradScaler(2**10)
            self.optimizer = torch.optim.Adam(self.radiance_field.parameters(), lr=1e-2, eps=1e-15, weight_decay=self.config["weight_decay"])
            self.scheduler = torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.01, total_iters=100),
                    torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[args.max_steps // 2, args.max_steps * 3 // 4, args.max_steps * 9 // 10,], gamma=0.33,),
                ])
        # calculate acc module.
        self.lpips_net = LPIPS(net="vgg").to(self.device)
        # print(self.test_dataset)

        # generate exp folder for save and load model.
        # self.exp_folder = os.path.join("./exp", )

    def lpips_norm_fn(self, x):
        return x[None, ...].permute(0, 3, 1, 2) * 2 - 1

    def lpips_fn(self, x, y):
        return self.lpips_net(self.lpips_norm_fn(x), self.lpips_norm_fn(y)).mean()

    def save_model(self):
        # save model into path.
        folder_path = self.exp_helper.exp_path
        torch.save(self.radiance_field.state_dict(), os.path.join(folder_path, "radiance_field.ckpt"))
        torch.save(self.occupancy_grid.state_dict(), os.path.join(folder_path, "occupancy_grid.ckpt"))
        print("Train result save at{}".format(folder_path))

    def load_model(self):
        folder_path = self.exp_helper.exp_path
        print("Load modle from {}".format(folder_path))
        self.radiance_field.load_state_dict(torch.load(os.path.join(folder_path, "radiance_field.ckpt")))
        self.occupancy_grid.load_state_dict(torch.load(os.path.join(folder_path, "occupancy_grid.ckpt")))
        # self.radiance_field.load_state_dict(torch.load("./ckpts/radiance_field_best.ckpt"))
        # self.occupancy_grid.load_state_dict(torch.load("./ckpts/occupancy_grid_best.ckpt"))

        # evaluation
        self.radiance_field.eval()
        self.occupancy_grid.eval()

    def train(self):
        # training
        if args.run_type != "train":
            print("please set --run_type = 'train'.")
            return
        tic = time.time()
        for step in range(args.max_steps + 1):
            self.radiance_field.train()

            i = torch.randint(0, len(self.train_dataset), (1,)).item()
            data = self.train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                density = self.radiance_field.query_density(x)
                return density * self.config["render_step_size"]

            # update occupancy grid
            self.occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn, occ_thre=1e-2,)

            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                self.radiance_field,
                self.occupancy_grid,
                rays,
                scene_aabb=self.scene_aabb,
                # rendering options
                near_plane=self.config["near_plane"],
                render_step_size=self.config["render_step_size"],
                render_bkgd=render_bkgd,
                cone_angle=self.config["cone_angle"],
                alpha_thre=self.config["alpha_thre"],
            )
            if n_rendering_samples == 0:
                continue

            if self.config["target_sample_batch_size"] > 0:
                # dynamic batch size for rays to keep sample batch size constant.
                num_rays = len(pixels)
                num_rays = int(
                    num_rays * (self.config["target_sample_batch_size"] / float(n_rendering_samples))
                )
                self.train_dataset.update_num_rays(num_rays)

            # compute loss
            loss = F.smooth_l1_loss(rgb, pixels)

            self.optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            self.grad_scaler.scale(loss).backward()
            self.optimizer.step()
            self.scheduler.step()

            if step % 200 == 0:
                elapsed_time = time.time() - tic
                loss = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(loss) / np.log(10.0)
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={loss:.5f} | psnr={psnr:.2f} | "
                    f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
                    f"max_depth={depth.max():.3f} | "
                )

        # save model into path.
        if args.save:
            self.save_model()

    def test(self):
        # ==========================================valid=============================================
        if args.run_type != "test":
            print("please set --run_type = 'test'.")
            return

        psnrs = []
        lpips = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(self.test_dataset))):
                data = self.test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image(
                    self.radiance_field,
                    self.occupancy_grid,
                    rays,
                    scene_aabb=self.scene_aabb,
                    # rendering options
                    near_plane=self.config["near_plane"],
                    render_step_size=self.config["render_step_size"],
                    render_bkgd=render_bkgd,
                    cone_angle=self.config["cone_angle"],
                    alpha_thre=self.config["alpha_thre"],
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                psnrs.append(psnr.item())
                lpips.append(self.lpips_fn(rgb, pixels).item())
                if args.save_image:
                    imageio.imwrite(os.path.join(self.exp_helper.rgb_path, "rgb_" + str(i) + ".png"), (rgb.cpu().numpy() * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(self.exp_helper.error_path, "error_" + str(i) + ".png"), ((rgb - pixels).norm(dim=-1).cpu().numpy() * 255).astype(np.uint8))
                    imageio.imwrite(os.path.join(self.exp_helper.depth_path, "depth_" + str(i) + ".png"), (depth * 255.0 / torch.max(depth)).cpu().numpy().astype(np.uint8))
        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")

    def predict(self, c2w):
        if args.run_type != "predict":
            print("please set --run_type = 'predict'.")
            return

        with torch.no_grad():
            self.test_dataset.OPENGL_CAMERA = False
            data = self.test_dataset.generate_data_from_pose(c2w)

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]

            # rendering
            rgb, acc, depth, _ = render_image(
                self.radiance_field,
                self.occupancy_grid,
                rays,
                scene_aabb=self.scene_aabb,
                # rendering options
                near_plane=self.config["near_plane"],
                render_step_size=self.config["render_step_size"],
                render_bkgd=render_bkgd,
                cone_angle=self.config["cone_angle"],
                alpha_thre=self.config["alpha_thre"],
                # test options
                test_chunk_size=args.test_chunk_size,
            )
        return rgb, acc, depth

    def get_config(self):
        config = {}
        if args.scene in MIPNERF360_UNBOUNDED_SCENES:
            # training parameters
            config["init_batch_size"] = 1024
            config["target_sample_batch_size"] = 1 << 18
            config["weight_decay"] = 0.0
            # scene parameters
            config["aabb"] = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=self.device)
            config["near_plane"] = 0.02
            config["far_plane"] = None
            # dataset parameters
            config["train_dataset_kwargs"] = {"color_bkgd_aug": "random", "factor": 4}
            config["test_dataset_kwargs"] = {"factor": 4}
            # model parameters
            config["grid_resolution"] = 128
            config["grid_nlvl"] = 4
            # render parameters
            config["render_step_size"] = 1e-3
            config["alpha_thre"] = 1e-2
            config["cone_angle"] = 0.004
        else:
            # training parameters
            config["init_batch_size"] = 1024
            config["target_sample_batch_size"] = 1 << 18
            config["weight_decay"] = (1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6)
            # scene parameters
            # aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
            # config["aabb"] = torch.tensor([-2.0, -2.0, -2.0, 2.0, 2.0, 2.0], device=self.device)
            config["aabb"] = torch.tensor([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0], device=self.device)
            config["near_plane"] = None
            config["far_plane"] = None
            # dataset parameters
            config["train_dataset_kwargs"] = {}
            config["test_dataset_kwargs"] = {}
            # model parameters
            config["grid_resolution"] = 128
            config["grid_nlvl"] = 1
            # render parameters
            config["render_step_size"] = 5e-3
            config["alpha_thre"] = 0.0
            config["cone_angle"] = 0.0
        self.config = config

    def run(self):
        if args.run_type == "train":
            self.train()
        elif args.run_type == "test":
            self.load_model()
            self.test()
        else:
            self.load_model()
            poses = torch.Tensor([
                [[-9.9990e-01, -4.1922e-03,  1.3346e-02, -7.6729e-03],
                 [-1.3989e-02,  2.9966e-01, -9.5394e-01,  1.4358e+00],
                 [-4.6566e-10, -9.5404e-01, -2.9969e-01,  3.7540e-01]],

                [[-9.3054e-01, -1.1708e-01,  3.4696e-01, -4.9283e-01],
                 [-3.6618e-01,  2.9751e-01, -8.8170e-01,  1.3307e+00],
                 [7.4506e-09, -9.4751e-01, -3.1972e-01,  4.0453e-01]]
            ]).to(self.device)

            c2w = torch.stack([poses[1]], dim=0)
            # c2w = torch.stack([self.test_dataset.camtoworlds[1]], dim=0)

            rgb, acc, depth = self.predict(c2w)
            imageio.imwrite("./ckpts/rgb_test.png", (rgb.cpu().numpy() * 255).astype(np.uint8))
            imageio.imwrite("./ckpts/depth_test.png", (depth.cpu().numpy() * 255).astype(np.uint8))


if __name__ == "__main__":
    args.max_steps = 20000
    args.img_h = 1920
    args.img_w = 1080
    args.scene = "ek"
    # args.scene = "man5"
    args.run_type = "test"
    mm = NerfNGP()
    mm.run()
