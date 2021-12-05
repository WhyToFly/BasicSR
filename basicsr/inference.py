import logging
import torch
from os import path as osp

import os

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
from basicsr.utils.options import dict2str, parse_options


def test_pipeline(root_path):
    inference_wav_dir = os.path.join(root_path, "inference_data")

    # parse options, set distributed setting, set random seed
    opt, _ = parse_options(root_path, is_train=False)

    torch.backends.cudnn.benchmark = True

    # create model
    model = build_model(opt)

    for filename in os.listdir(inference_wav_dir):
        if filename.endswith(".flac"):
            print("Inference for " + filename)
            model.inference(inference_wav_dir, filename)


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
