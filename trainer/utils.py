import os
import random
import time
import datetime
import sys
from pathlib import Path

import cv2
import yaml
from torch.autograd import Variable
import torch
from visdom import Visdom
import torch.nn.functional as F
import numpy as np


class Resize():
    def __init__(self, size_tuple, use_cv=True):
        self.size_tuple = size_tuple
        self.use_cv = use_cv

    def __call__(self, tensor):
        """
            Resized the tensor to the specific size

            Arg:    tensor  - The torch.Tensor obj whose rank is 4
            Ret:    Resized tensor
        """
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=[self.size_tuple[0], self.size_tuple[1]])
        tensor = tensor.squeeze(0)

        return tensor  # 1, 64, 128, 128


class ToTensor():
    def __call__(self, tensor):
        """
        tensor: H W C
        target: C H W
        """
        if len(tensor.shape) == 2:
            tensor = np.expand_dims(tensor, 0)
            # tensor = np.array(tensor)
            return torch.from_numpy(tensor)  # C H W
        elif len(tensor.shape) == 3:
            return torch.from_numpy(tensor.transpose(2, 0, 1))  # C H W


def tensor2image(tensor):
    """
    @param tensor: (1, 3, 256, 256)
    @return:
    """
    #########################
    # tensor: C H W --> H W C (RGB)
    #########################
    image_numpy = tensor[0].cpu().float().numpy()  # convert it into a numpy array
    if image_numpy.shape[0] == 1:  # grayscale to RGB
        image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = ((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2) * 255.0
    else:
        image_numpy = ((np.transpose(image_numpy, (1, 2, 0)) + 1) / 2) * 255.0  # post-processing: tranpose and scaling
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(np.uint8)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    cv2.imwrite(image_path, image_numpy)


class Logger():
    def __init__(self, config, batches_epoch, start_epoch_continue, end_epoch_continue):
        self.viz = Visdom(env=config['env_name'], port=config['display_port'])
        self.n_epochs = config['n_epochs'] + end_epoch_continue
        self.batches_epoch = batches_epoch
        self.epoch = 1 + start_epoch_continue
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.metric_windows = {}
        self.image_windows = {}
        self.distance_windows = {}
        self.lr_windows = {}

        self.config = config
        self.env_name = config['name']
        path_t = self.config['save_root'] + self.config['run_name'] + 'images'
        # print(path_t)
        self.img_dir = Path(os.getcwd()).as_posix() + path_t
        # print(self.img_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)

    def plot_metrics(self, metrics, epoch):
        if not hasattr(self, 'metrics_plot'):
            self.metrics_plot = {}
        for metric_name, metric in metrics.items():
            if metric_name != 'epoch':
                self.metrics_plot[metric_name] = metric
        for metric_name, metric in self.metrics_plot.items():
            if metric_name not in self.metric_windows:
                self.metric_windows[metric_name] = self.viz.line(X=np.array([epoch]),
                                                                 Y=np.array([metric]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': metric_name,
                                                                       'title': metric_name})
            else:
                self.viz.line(X=np.array([epoch]), Y=np.array([metric]),
                              win=self.metric_windows[metric_name], update='append')
            # Reset metrics_plot for next epoch
            # self.metrics_plot[metric_name] = 0.0

    def plot_distances(self, distances, epoch):
        for distance_name, distance in distances.items():
            if distance_name not in self.distance_windows:
                self.distance_windows[distance_name] = self.viz.line(X=np.array([epoch]),
                                                                     Y=np.array([distance]),
                                                                     opts={'xlabel': 'epochs', 'ylabel': distance_name,
                                                                           'title': distance_name})
            else:
                self.viz.line(X=np.array([epoch]), Y=np.array([distance]),
                              win=self.distance_windows[distance_name], update='append')

    # plot the learning rate value during training
    def plot_lrs(self, lrs, epoch):
        if not hasattr(self, 'lrs_plot'):
            self.lrs_plot = {}
        for lr_name, lr in lrs.items():
            self.lrs_plot[lr_name] = lr
        for lr_name, lr in self.lrs_plot.items():
            if lr_name not in self.lr_windows:
                self.lr_windows[lr_name] = self.viz.line(X=np.array([epoch]),
                                                         Y=np.array([lr]),
                                                         opts={'xlabel': 'epochs', 'ylabel': lr_name,
                                                               'title': lr_name})
            else:
                self.viz.line(X=np.array([epoch]), Y=np.array([lr]),
                              win=self.lr_windows[lr_name], update='append')
            # Reset metrics_plot for next epoch
            # self.metrics_plot[metric_name] = 0.0

    def log(self, losses=None, images=None, do_plot=False):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            if loss_name not in self.losses:
                self.losses[loss_name] = losses[loss_name].item()
            else:
                self.losses[loss_name] += losses[loss_name].item()

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))
        if do_plot:
            # Draw images
            for image_name, tensor in images.items():
                # H W C-->C H W
                img = tensor2image(tensor.data).transpose(2, 0, 1)
                # img = tensor2image(tensor.data)
                if image_name not in self.image_windows:
                    self.image_windows[image_name] = self.viz.image(img, opts={'title': image_name})
                else:
                    self.viz.image(img, win=self.image_windows[image_name],
                                   opts={'title': image_name})

        # End of epoch
        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():

                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')


        else:
            self.batch += 1


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    # print ('m:',m)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def smooothing_loss(y_pred):
    dy_1 = y_pred[:, :, 1:, :]
    dy_2 = y_pred[:, :, :-1, :]
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = dx * dx
    dy = dy * dy
    d = torch.mean(dx) + torch.mean(dy)
    grad = d
    return d
