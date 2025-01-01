import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import random
import os
from skimage.transform import resize

class MyDataset(Dataset):
    def __init__(self, mode, data_dict, deactivate_random = False):
        self.data_dict = data_dict
        self.mode = mode
        if mode == "training":
            self.input_300k_path = data_dict["base_dir_dobs_300k_train"]
            self.input_400k_path = data_dict["base_dir_dobs_400k_train"]
            self.input_500k_path = data_dict["base_dir_dobs_500k_train"]
            self.speed_path = data_dict["base_dir_speed_train"]
        elif mode == "validation": 
            self.input_300k_path = data_dict["base_dir_dobs_300k_eval"]
            self.input_400k_path = data_dict["base_dir_dobs_400k_eval"]
            self.input_500k_path = data_dict["base_dir_dobs_500k_eval"]
            self.speed_path = data_dict["base_dir_speed_eval"]
        else:
            raise ValueError("Wrong!")

        self.length = len(os.listdir(self.input_300k_path))
        self.deactivate_random =  deactivate_random 
        self.noise_level = data_dict["noise_level"]
        self.ratio = data_dict["input_ratio"]
        self.device = data_dict["device"]

        self.max_output = data_dict["max_output"]
        self.min_output = data_dict["min_output"]
        # self.max_input = data_dict["max_input"]
        # self.min_input = data_dict["min_input"]

    def __len__(self):
        return self.length
    
    def add_awgn(self, signal, snr_dB):
        signal_power = np.mean(signal**2)

        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * torch.randn_like(torch.tensor(signal)).numpy()
        noisy_signal = signal + noise
        return noisy_signal
    
    def denormalize(self, pred_out):
        return (self.max_output - self.min_output) * (pred_out + torch.tensor(1., device=self.device)) / 2 + self.min_output

    def __getitem__(self, index):
        if self.mode == "training":
            inp_300k = np.load(os.path.join(self.input_300k_path, "train_" + str(index+1) + ".npy"))
            real_300k, imag_300k = inp_300k.real, inp_300k.imag
            inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256
            inputs_300k = np.stack([inputs_300k[:, :, self.ratio * kk] for kk in range(inputs_300k.shape[1] // self.ratio)], axis=-1)  # 2,256,32
            
            inp_400k = np.load(os.path.join(self.input_400k_path, "train_" + str(index+1) + ".npy"))
            real_400k, imag_400k = inp_400k.real, inp_400k.imag
            inputs_400k = np.stack((real_400k, imag_400k))  # 2,256,256
            inputs_400k = np.stack([inputs_400k[:, :, self.ratio * kk] for kk in range(inputs_400k.shape[1] // self.ratio)], axis=-1)  # 2,256,32

            inp_500k = np.load(os.path.join(self.input_500k_path, "train_" + str(index+1) + ".npy"))
            real_500k, imag_500k = inp_500k.real, inp_500k.imag
            inputs_500k = np.stack((real_500k, imag_500k))  # 2,256,256
            inputs_500k = np.stack([inputs_500k[:, :, self.ratio * kk] for kk in range(inputs_500k.shape[1] // self.ratio)], axis=-1)  # 2,256,32
            
            inputs = np.stack([inputs_300k,inputs_400k,inputs_500k], axis= 0)
            output_full = np.load(os.path.join(self.speed_path, "train_" + str(index+1) + ".npy"))

        elif self.mode == "validation":
            inp_300k = np.load(os.path.join(self.input_300k_path, "train_" + str(index+6601) + ".npy"))
            real_300k, imag_300k = inp_300k.real, inp_300k.imag
            inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256
            inputs_300k = np.stack([inputs_300k[:, :, self.ratio * kk] for kk in range(inputs_300k.shape[1] // self.ratio)], axis=-1)  # 2,256,32
            
            inp_400k = np.load(os.path.join(self.input_400k_path, "train_" + str(index+6601)+ ".npy"))
            real_400k, imag_400k = inp_400k.real, inp_400k.imag
            inputs_400k = np.stack((real_400k, imag_400k))  # 2,256,256
            inputs_400k = np.stack([inputs_400k[:, :, self.ratio * kk] for kk in range(inputs_400k.shape[1] // self.ratio)], axis=-1)  # 2,256,32

            inp_500k = np.load(os.path.join(self.input_500k_path, "train_" + str(index+6601) + ".npy"))
            real_500k, imag_500k = inp_500k.real, inp_500k.imag
            inputs_500k = np.stack((real_500k, imag_500k))  # 2,256,256
            inputs_500k = np.stack([inputs_500k[:, :, self.ratio * kk] for kk in range(inputs_500k.shape[1] // self.ratio)], axis=-1)  # 2,256,32
            
            inputs = np.stack([inputs_300k,inputs_400k,inputs_500k], axis= 0)
            output_full = np.load(os.path.join(self.speed_path, "train_" + str(index+6601) + ".npy"))

        output = resize(output_full[90:390,90:390], self.data_dict["resize_size"], mode='reflect', anti_aliasing=True) - self.data_dict["output_background"]# 480 中心 [90:390, 90:390] 子范围

        # Add Noise for Training and Sampling
        if not self.deactivate_random: 
            random_number = random.random()
            if random_number < 0.33:
                inputs = self.add_awgn(inputs,10)
            elif random_number < 0.66:
                inputs = self.add_awgn(inputs,5)
            else:
                inputs = inputs
        else:
            if self.noise_level == "free":
                inputs = inputs
            elif self.noise_level == "10db":
                inputs = self.add_awgn(inputs,10)
            elif self.noise_level == "5db":
                inputs = self.add_awgn(inputs,5)
            else: 
                raise ValueError("Unknown Noise Type")
        
        if self.data_dict["use_mask"]:
            inputs = inputs * self.data_dict["mask"]
            
        # # Log_rescale
        # inputs = 2 * (inputs - self.min_input)/ (self.max_input - self.min_input) - 1.
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        
        inputs = torch.tensor(inputs, dtype=torch.float32).view((6, 256, 256))
        output = torch.tensor(output, dtype=torch.float32)

        return inputs, output


class MyDataset_300k(Dataset):
    def __init__(self, mode, data_dict, deactivate_random = False):
        self.data_dict = data_dict
        self.mode = mode
        if mode == "training":
            self.input_300k_path = data_dict["base_dir_dobs_300k_train"]
            self.speed_path = data_dict["base_dir_speed_train"]
        elif mode == "validation": 
            self.input_300k_path = data_dict["base_dir_dobs_300k_eval"]
            self.speed_path = data_dict["base_dir_speed_eval"]
        else:
            raise ValueError("Wrong!")

        self.length = len(os.listdir(self.input_300k_path))
        self.deactivate_random =  deactivate_random 
        self.ratio = data_dict["input_ratio"]
        self.device = data_dict["device"]

        self.max_output = data_dict["max_output"]
        self.min_output = data_dict["min_output"]

    def __len__(self):
        return self.length
    
    def add_awgn(self, signal, snr_dB):
        signal_power = np.mean(signal**2)

        snr_linear = 10**(snr_dB / 10.0)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * torch.randn_like(torch.tensor(signal)).numpy()
        noisy_signal = signal + noise
        return noisy_signal
    
    def denormalize(self, pred_out):
        return (self.max_output - self.min_output) * (pred_out + torch.tensor(1., device=self.device)) / 2 + self.min_output

    def __getitem__(self, index):
        if self.mode == "training":
            inp_300k = np.load(os.path.join(self.input_300k_path, "train_" + str(index+1) + ".npy"))
            real_300k, imag_300k = inp_300k.real, inp_300k.imag
            inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256

            inputs = np.stack([inputs_300k], axis= 0)
            output_full = np.load(os.path.join(self.speed_path, "train_" + str(index+1) + ".npy"))

        elif self.mode == "validation":
            inp_300k = np.load(os.path.join(self.input_300k_path, "train_" + str(index+6601) + ".npy"))
            real_300k, imag_300k = inp_300k.real, inp_300k.imag
            inputs_300k = np.stack((real_300k, imag_300k))  # 2,256,256
            
            inputs = np.stack([inputs_300k], axis= 0)
            output_full = np.load(os.path.join(self.speed_path, "train_" + str(index+6601) + ".npy"))

        # Add Noise for Training and Sampling
        if not self.deactivate_random: 
            random_number = random.random()
            if random_number < 0.33:
                inputs = self.add_awgn(inputs,10)
            elif random_number < 0.66:
                inputs = self.add_awgn(inputs,5)
            else:
                inputs = inputs
        else:
            inputs_free = inputs
            inputs_10db = self.add_awgn(inputs,10)
            inputs_5db = self.add_awgn(inputs,5)
            
        if self.data_dict["use_mask"]:
            inputs_free = inputs_free * self.data_dict["mask"]
            inputs_10db = inputs_10db * self.data_dict["mask"]
            inputs_5db = inputs_5db * self.data_dict["mask"]

        # 找到不能被 self_ratio 整除的行和列
        invalid_rows = [i for i in range(inputs.shape[1]) if i % self.ratio != 0]
        invalid_cols = [j for j in range(inputs.shape[2]) if j % self.ratio != 0]
            
        # 将不能整除的行和列置为 0
        inputs_free[:, invalid_rows, :] = 0  # 置为 0 的行
        inputs_free[:, :, invalid_cols] = 0  # 置为 0 的列
        inputs_10db[:, invalid_rows, :] = 0  # 置为 0 的行
        inputs_10db[:, :, invalid_cols] = 0  # 置为 0 的列
        inputs_5db[:, invalid_rows, :] = 0  # 置为 0 的行
        inputs_5db[:, :, invalid_cols] = 0  # 置为 0 的列

        inputs_free = torch.tensor(inputs_free, dtype=torch.float32).view((2, 256, 256))
        inputs_10db = torch.tensor(inputs_10db, dtype=torch.float32).view((2, 256, 256))
        inputs_5db = torch.tensor(inputs_5db, dtype=torch.float32).view((2, 256, 256))
        
        output = resize(output_full[90:390,90:390], self.data_dict["resize_size"], mode='reflect', anti_aliasing=True) - self.data_dict["output_background"]# 480 中心 [90:390, 90:390] 子范围
        output = 2 * (output - self.min_output) / (self.max_output - self.min_output) - 1.
        output = torch.tensor(output, dtype=torch.float32)
        
        return inputs_free, inputs_10db, inputs_5db, output
