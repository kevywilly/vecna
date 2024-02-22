import numpy as np
import torch

from src.kinematics import forward
from src.training.functional import normalize_max
from src.vectors import Angle3


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = []
        for c in range (-90,90,10):
            for f in range(-90,90,10):
                for t in range(-90,90,10):
                    x,y,z = forward(Angle3(c,f,t).numpy())
                    self.items.append([round(x,2), round(y,2), round(z,2), c/90.0, f/90.0, t/90.0])

        self.items = np.array(self.items)
        X = self.items[:,0:3]
        Y = self.items[:,3:6]
        self.maxs = np.max(X,0)
        self.means = np.mean(X,0)
        self.stds = np.std(X,0)
        X = normalize_max(X, self.maxs)
        self.inputs = torch.from_numpy(X.astype(np.float32))
        self.targets = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        return self.inputs[idx].reshape(1,-1), self.targets[idx].reshape(1,-1)