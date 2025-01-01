# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import random

import torchvision.transforms as transforms
import torch.utils.data as data


import glob
from PIL import Image
from skimage.transform import resize


def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def get_dataset_pytorch(config):
  """
  Create data loaders for training and evaluation. But using the torch.Dataset
  """

  if config.data.dataset == 'AI4Scup2':
    file = {
    "use_mask": False,
    #"mask": np.load("/home/caoxiang/Desktop/full_waveform_inversion/mask.npy"),
    "max_input" : 8882.412, #from training dataset statistics 
    "min_input" : -7292.7476,
    "max_output" : 1595.1279,
    "min_output" : 1408.692,
    "resize_size": (config.data.image_size, config.data.image_size),#rough 120, full 300
    }

    file["base_dir_speed_NIO_5db_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_5db/train"
    file["base_dir_speed_NIO_5db_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_5db/eval"
    file["base_dir_speed_NIO_10db_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_10db/train"
    file["base_dir_speed_NIO_10db_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_10db/eval"
    file["base_dir_speed_NIO_free_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_free/train"
    file["base_dir_speed_NIO_free_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed_NIO_free/eval"
    
    file["base_dir_speed_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed/train"
    file["base_dir_speed_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2/speed/eval"
    file["device"] = config.device

    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, mode, data_dict):
          self.data_dict, self.mode = data_dict, mode 
          if self.mode == "training":
            self.speed_path = data_dict["base_dir_speed_train"]
            self.speed_NIO_5db_path = data_dict["base_dir_speed_NIO_5db_train"]
            self.speed_NIO_10db_path = data_dict["base_dir_speed_NIO_10db_train"]
            self.speed_NIO_free_path = data_dict["base_dir_speed_NIO_free_train"]
          elif self.mode == "validation": 
            self.speed_path = data_dict["base_dir_speed_eval"]
            self.speed_NIO_5db_path = data_dict["base_dir_speed_NIO_5db_eval"]
            self.speed_NIO_10db_path = data_dict["base_dir_speed_NIO_10db_eval"]
            self.speed_NIO_free_path = data_dict["base_dir_speed_NIO_free_eval"]
          else:
              raise ValueError("Wrong!")

          self.max_output = data_dict["max_output"]
          self.min_output = data_dict["min_output"]
          self.max_input = data_dict["max_input"]
          self.min_input = data_dict["min_input"]
          self.resize_size = self.data_dict["resize_size"]
          self.device = data_dict["device"]

      def __len__(self):
          return len(os.listdir(self.speed_path))

      def denormalize(self, pred_out):
        return (self.max_output - self.min_output) * (pred_out + torch.tensor(1., device=self.device)) / 2 + self.min_output
      
      def log_image(self, path, reference_field, pred_field, field):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 用于确保统一的colorbar范围
        vmin, vmax = self.min_output, self.max_output

        # 显示第二个图像 pred_field
        im0 = axes[0].imshow(reference_field[0, :, :], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[0].set_title("Reference field")
        fig.colorbar(im0, ax=axes[0])

        # 显示第二个图像 pred_field
        im1 = axes[1].imshow(pred_field[0, :, :], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[1].set_title("Predicted field")
        fig.colorbar(im1, ax=axes[1])

         # 显示第一个图像 field
        im2 = axes[2].imshow(field[0, :, :], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[2].set_title("Field")
        fig.colorbar(im2, ax=axes[2])

        # 调整布局并保存图像
        plt.tight_layout()
        plt.savefig(path)
        plt.show()
        plt.close()

      def __getitem__(self, index):
        if self.mode == "training":
          speed_full = np.load(os.path.join(self.speed_path, "train_" + str(index+1) + ".npy"))
          speed_NIO_5db_full = np.load(os.path.join(self.speed_NIO_5db_path, "train_" + str(index+1) + ".npy"))
          speed_NIO_10db_full = np.load(os.path.join(self.speed_NIO_10db_path, "train_" + str(index+1) + ".npy"))
          speed_NIO_free_full = np.load(os.path.join(self.speed_NIO_free_path, "train_" + str(index+1) + ".npy"))

        elif self.mode == "validation":
          speed_full = np.load(os.path.join(self.speed_path, "train_" + str(index+6601) + ".npy"))
          speed_NIO_5db_full = np.load(os.path.join(self.speed_NIO_5db_path, "train_" + str(index+6601) + ".npy"))
          speed_NIO_10db_full = np.load(os.path.join(self.speed_NIO_10db_path, "train_" + str(index+6601) + ".npy"))
          speed_NIO_free_full = np.load(os.path.join(self.speed_NIO_free_path, "train_" + str(index+6601) + ".npy"))

        # Resize to 256*256
        speed = resize(speed_full[90:390,90:390], self.resize_size, mode='reflect', anti_aliasing=True)# 480 中心 [90:390, 90:390] 子范围
        speed_5db_NIO = resize(speed_NIO_5db_full[90:390,90:390], self.resize_size, mode='reflect', anti_aliasing=True)# 480 中心 [90:390, 90:390] 子范围
        speed_10db_NIO = resize(speed_NIO_10db_full[90:390,90:390], self.resize_size, mode='reflect', anti_aliasing=True) # 480 中心 [90:390, 90:390] 子范围
        speed_free_NIO = resize(speed_NIO_free_full[90:390,90:390], self.resize_size, mode='reflect', anti_aliasing=True) # 480 中心 [90:390, 90:390] 子范围

        # Normalize to [-1,1]
        speed = 2 * (speed.reshape(1, self.resize_size[0], self.resize_size[1]) - self.min_output) / (self.max_output - self.min_output) - 1.
        speed_5db_NIO = 2 * (speed_5db_NIO.reshape(1, self.resize_size[0], self.resize_size[1]) - self.min_output) / (self.max_output - self.min_output) - 1.
        speed_10db_NIO = 2 * (speed_10db_NIO.reshape(1, self.resize_size[0], self.resize_size[1]) - self.min_output) / (self.max_output - self.min_output) - 1.
        speed_free_NIO = 2 * (speed_free_NIO.reshape(1, self.resize_size[0], self.resize_size[1]) - self.min_output) / (self.max_output - self.min_output) - 1.
        
        return torch.tensor(speed, dtype=torch.float32), torch.tensor(speed_5db_NIO, dtype=torch.float32), torch.tensor(speed_10db_NIO, dtype=torch.float32), torch.tensor(speed_free_NIO, dtype=torch.float32)

    train_dataset = Create_Custom_Dataset("training", file)
    train_loader = data.DataLoader(train_dataset, shuffle = True, batch_size = config.training.batch_size, num_workers = 8)
    eval_dataset = Create_Custom_Dataset("validation", file)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = config.eval.batch_size, num_workers = 4)
  
  elif config.data.dataset == 'AI4Scup2_speed':
    file = {
        "max_output" : 1595.1279,
        "min_output" : 1408.692,
        "resize_size": (config.data.image_size, config.data.image_size),
        }

    file["base_dir_speed_train"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/train"
    file["base_dir_speed_eval"] = "/home/caoxiang/Desktop/Datasets/AI4Scup2_simulated/speed/eval"
    file["device"] = config.device
    
    class Create_Custom_Dataset(data.Dataset):
      def __init__(self, mode, data_dict):
          self.data_dict, self.mode = data_dict, mode 
          if self.mode == "training":
            self.speed_path = data_dict["base_dir_speed_train"]
          elif self.mode == "validation": 
            self.speed_path = data_dict["base_dir_speed_eval"]
          else:
              raise ValueError("Wrong!")

          self.max_output = data_dict["max_output"]
          self.min_output = data_dict["min_output"]
          self.resize_size = self.data_dict["resize_size"]
          self.device = data_dict["device"]

      def __len__(self):
          return len(os.listdir(self.speed_path))

      def denormalize(self, pred_out):
        return (self.max_output - self.min_output) * (pred_out + torch.tensor(1., device=self.device)) / 2 + self.min_output
      
      def __getitem__(self, index):
        if self.mode == "training":
          speed_full = np.load(os.path.join(self.speed_path, "train_" + str(index+1) + ".npy"))

        elif self.mode == "validation":
          speed_full = np.load(os.path.join(self.speed_path, "train_" + str(index+6601) + ".npy"))

        speed = resize(speed_full[90:390,90:390], self.resize_size, mode='reflect', anti_aliasing=True)
        speed = 2 * (speed.reshape(1, self.resize_size[0], self.resize_size[1]) - self.min_output) / (self.max_output - self.min_output) - 1.

        return torch.tensor(speed, dtype=torch.float32)
    
    train_dataset = Create_Custom_Dataset("training", file)
    train_loader = data.DataLoader(train_dataset, shuffle = False, batch_size = 1, num_workers = 1)
    eval_dataset = Create_Custom_Dataset("validation", file)
    eval_loader = data.DataLoader(eval_dataset, shuffle = False, batch_size = 1, num_workers = 1)
  

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')
  
  return train_dataset, train_loader, eval_dataset, eval_loader