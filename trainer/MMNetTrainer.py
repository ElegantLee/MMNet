#!/usr/bin/python3

from pathlib import Path
import torch.nn
from torch.utils.data import DataLoader
from Model.CycleGan import *
from .utils import LambdaLR
from .datasets import ValDataset
from .utils import Resize, ToTensor
from .reg import Reg
from .transformer import Transformer_2D
import numpy as np
import cv2
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class MMNet_Trainer():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.path_save = os.path.join(os.getcwd()) + self.config['save_root'] + self.config['run_name']

        # define networks
        self.netG_A2B = Generator(config['input_nc'], config['output_nc']).cuda()
        self.optimizer_G = torch.optim.Adam(self.netG_A2B.parameters(), lr=config['g_lr'], betas=(0.5, 0.999))
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(config['n_epochs'], config['epoch'], config['decay_epoch']).step
        )

        if config['regist']:
            self.R_A = Reg(config['input_nc'], config['output_nc']).cuda()
            self.spatial_transform = Transformer_2D().cuda()

            self.optimizer_R_A = torch.optim.Adam(self.R_A.parameters(), lr=config['r_lr'], betas=(0.5, 0.999))
            self.lr_scheduler_R_A = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer_R_A, lr_lambda=LambdaLR(config['n_epochs'], config['epoch'], config['decay_epoch']).step
            )

        # Inputs & targets memory allocation
        self.Tensor = torch.cuda.FloatTensor if config['cuda'] else torch.Tensor

        val_transforms = [
            ToTensor(),
            Resize(size_tuple=(config['size'], config['size']))]

        self.val_data = DataLoader(ValDataset(config['val_dataroot'], transforms_=val_transforms, unaligned=False),
                                   batch_size=config['val_batchSize'], shuffle=False, num_workers=config['n_cpu'])

    def test(self, ):
        # define the path
        result_path = Path(os.getcwd()).as_posix() + self.config['image_save'] + self.config['run_name']
        images_path = Path(os.path.join(result_path, 'images')).as_posix()
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        path_save = os.path.join(os.getcwd()) + self.config['save_root'] + self.config['run_name']
        self.netG_A2B.load_state_dict(torch.load(Path(os.path.join(path_save, 'netG_A2B.pth')).as_posix()))

        with torch.no_grad():
            num = 0
            for batch in enumerate(tqdm(self.val_data)):
                num += 1
                real_A = batch['A_img'].cuda()
                real_B = batch['B_img'].cuda().detach().cpu().numpy().squeeze()

                fake_B = self.netG_A2B(real_A)  # .detach().cpu().numpy().squeeze()

                fake_B = fake_B.detach().cpu().numpy().squeeze()

                real_A = ((real_A.detach().cpu().numpy().squeeze() + 1) / 2) * 255
                real_B = ((real_B + 1) / 2) * 255
                fake_B = 255 * ((fake_B + 1) / 2)

                if self.config['input_nc'] != 1:
                    fake_B = np.transpose(fake_B, (1, 2, 0))
                    real_B = np.transpose(real_B, (1, 2, 0))
                    real_A = np.transpose(real_A, (1, 2, 0))

                cv2.imwrite(result_path + '/images/' + str(num) + '_real_A.png', real_A)
                cv2.imwrite(result_path + '/images/' + str(num) + '_real_B.png', real_B)
                cv2.imwrite(result_path + '/images/' + str(num) + '_fake_B.png', fake_B)