#!/usr/bin/python3

import argparse
import os
from trainer import MMNet_Trainer
import yaml

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Yaml/MMNet.yaml', help='Path to the config file.')
    opts = parser.parse_args()
    config = get_config(opts.config)

    trainer = MMNet_Trainer(config)
    trainer.test()


###################################
if __name__ == '__main__':
    main()
