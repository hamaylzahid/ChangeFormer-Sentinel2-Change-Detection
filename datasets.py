# put dataset class here
import torch
from torch.utils.data import Dataset
from utils import augment_patch
import numpy as np

class ChangeDataset(Dataset):
    def __init__(self, patches1, patches2, ndvi_thresh=0.01, augment=False):
        assert len(patches1) == len(patches2)
        self.data = []
        self.augment = augment
        for t1, t2 in zip(patches1, patches2):
            change = np.abs(t2[:,:,4] - t1[:,:,4])
            if np.max(change) > ndvi_thresh:
                mask = (change>0.05).astype(np.float32)
                self.data.append((t1,t2,mask))
        if len(self.data)==0:
            raise ValueError("No meaningful patches. Reduce threshold or patch size.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        t1, t2, mask = self.data[idx]
        if self.augment:
            t1, t2, mask = augment_patch(t1,t2,mask)
        t1 = torch.tensor(t1).permute(2,0,1)
        t2 = torch.tensor(t2).permute(2,0,1)
        mask = torch.tensor(mask).unsqueeze(0)
        return t1, t2, mask