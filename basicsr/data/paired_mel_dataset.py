from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import pickle
import os
import torch

@DATASET_REGISTRY.register()
class PairedMelDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedMelDataset, self).__init__()
        self.opt = opt
        self.io_backend_opt = opt['io_backend']

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []
        
        for filename in os.listdir(self.gt_folder):
            if filename.endswith(".pkl"):
                self.paths.append({"gt_path": os.path.join(self.gt_folder, filename), \
                    "lq_path": os.path.join(self.lq_folder, filename)})
        

    def __getitem__(self, index):

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC
        gt_path = self.paths[index]['gt_path']
        img_gt = pickle.load(open(gt_path, "rb"))[:,:,None]
        
        # use last 172 mels as input
        #img_gt = img_gt[-172:]
        
        lq_path = self.paths[index]['lq_path']
        img_lq = pickle.load(open(lq_path, "rb"))[:,:,None]
        
        # use first 344 mels as input
        #img_lq = img_lq[:344]
        
        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], rotation=False, hflip=self.opt['use_rot'])


        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]
            # crop to something divisible by 4 (needed by pixel_unshuffle)
            img_gt = img_gt[:, :img_gt.shape[1] - img_gt.shape[1]%4, :]
            img_lq = img_lq[:, :img_lq.shape[1] - img_lq.shape[1]%4, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        #img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        
        # convert to tensor
        img_gt = torch.Tensor(img_gt)
        img_lq = torch.Tensor(img_lq)
        
        
        # convert HWC to CHW
        img_gt = torch.permute(img_gt, [2, 0, 1])
        img_lq = torch.permute(img_lq, [2, 0, 1])
        
        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
