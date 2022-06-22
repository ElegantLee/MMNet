import glob
import random
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path


##################BraTs2015######################
# class ValDataset(Dataset):
#     def __init__(self, root,count = None,transforms_=None, unaligned=False):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#         root_A = Path(os.path.join(os.getcwd(), "%s/A/val_A.npy" % root)).as_posix()
#         root_B = Path(os.path.join(os.getcwd(), "%s/B/val_B.npy" % root)).as_posix()
#         self.files_A = np.load(root_A).astype(np.float32)
#         self.files_B = np.load(root_B).astype(np.float32)
#
#     def __getitem__(self, index):
#         item_A = self.transform(self.files_A[index % self.files_A.shape[0]])
#         if self.unaligned:
#             item_B = self.transform(self.files_B[random.randint(0, self.files_B.shape[0] - 1)])
#         else:
#             item_B = self.transform(self.files_B[index % self.files_B.shape[0]])
#         return {'A_img': item_A, 'B_img': item_B}
#     def __len__(self):
#         return max(self.files_A.shape[0], self.files_B.shape[0])


####################OASIS3######################
class ValDataset(Dataset):
    def __init__(self, root, count=None, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob("%s/A/*" % root))
        self.files_B = sorted(glob.glob("%s/B/*" % root))

    def __getitem__(self, index):
        item_A = self.transform(np.load(self.files_A[index % len(self.files_A)]).astype(np.float32))
        if self.unaligned:
            item_B = self.transform(np.load(self.files_B[random.randint(0, len(self.files_B) - 1)]))
        else:
            item_B = self.transform(np.load(self.files_B[index % len(self.files_B)]).astype(np.float32))
        return {'A_img': item_A, 'B_img': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
