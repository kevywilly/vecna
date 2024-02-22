import math

from src.vecna import Vecna
import torch
import torch.nn as nn
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = []
        for c in range (-90,90,10):
            for f in range(-90,90,10):
                for t in range(-90,90,10):
                    x,y,z = Kinematics.forward(Vector3(c,f,t)).array()
                    self.items.append([round(x,2), round(y,2), round(z,2), c/90.0, f/90.0, t/90.0])

        self.items = np.array(self.items)
        X = self.items[:,0:3]
        Y = self.items[:,3:6]
        self.maxs = np.max(X,0)
        self.means = np.mean(X,0)
        self.stds = np.std(X,0)
        X = self.normalize(X, self.means, self.stds, self.maxs)
        self.inputs = torch.from_numpy(X.astype(np.float32))
        #self.inputs = torch.nn.functional.normalize(torch.from_numpy(X.astype(np.float32)),dim=0)
        self.targets = torch.from_numpy(Y.astype(np.float32))

        #X = (X-self.means)/self.stds
        #self.X = torch.nn.functional.normalize(input=torch.from_numpy(X.astype(np.float32)))
        # self.X = torch.from_numpy(X.astype(np.float32))
        # self.Y = torch.from_numpy(Y.astype(np.float32))

    @staticmethod
    def normalize(x, means, stds, maxs):
        return (x/np.max(maxs))
        #return (x-means)/stds

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        return self.inputs[idx].reshape(1,-1), self.targets[idx].reshape(1,-1)

model = nn.Sequential(
    nn.Linear(3,16),
    nn.Linear(16,32),
    nn.Linear(32,64),
    nn.ReLU(),
    nn.Linear(64,3)
)

best_model = '/home/zero1/vecna/best_model.pth'

print('loading model')
model.load_state_dict(torch.load(best_model))
print('done')

model.eval()

class Vector3:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def array(self):
        return [self.x, self.y, self.z]

    def numpy(self):
        return np.array(self.array())

    def __repr__(self):
        return f'({self.x},{self.y},{self.z})'


class Kinematics:
    @classmethod
    def forward(cls, angles: Vector3):
        v = angles.array()
        theta0,theta1,theta2 = angles.numpy() - Vecna.dims.thetas

        hyp0 = math.cos(math.radians(theta0)) * Vecna.dims.coxa
        x0 = math.sin(math.radians(theta0)) * Vecna.dims.coxa
        z0 = 0

        x1 = 0
        hyp1 = math.cos(math.radians(theta1)) * Vecna.dims.femur
        z1 = math.sin(math.radians(theta1)) * Vecna.dims.femur

        hyp2 = math.cos(math.radians(theta2)) * Vecna.dims.tibia

        y = hyp0+hyp1+hyp2
        x = math.sin(math.radians(theta0)) * y
        z = z0+z1

        return Vector3(int(math.floor(x)), int(math.floor(y)), int(math.floor(z)))

    @classmethod
    def inverse(cls, position: Vector3):
        x = position.numpy()
        x = Dataset.normalize(x, Vecna.training.means, Vecna.training.stds, Vecna.training.maxs)
        input = torch.from_numpy(x.astype(np.float32)).reshape(1,-1)
        output = model.forward(input)
        angles = np.floor(output.detach().numpy()*90)[0]
        return Vector3(int(angles[0]),int(angles[1]),int(angles[2]))


