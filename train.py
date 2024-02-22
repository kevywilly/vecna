from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import src.training.models as models
from src.training.datasets import Dataset
from src.training.functional import normalize_max


dataset = Dataset()
test_percent = 0.2
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=100,
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=100,
                                          shuffle=True,
                                          num_workers=0)

model = models.Kinematics3(pretrained=False)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 1500

train_losses = np.zeros(epochs)
test_losses = np.zeros(epochs)
best_loss = 1e9

for it in range(epochs):
    # zero the parameter gradients
    model.train()
    t0 = datetime.now()

    optimizer.zero_grad()
    train_loss = []

    for inputs, targets in iter(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_loss.append(loss.item())
        # Backward and optimize
        loss.backward()
        optimizer.step()

    train_loss = np.mean(train_loss)
    train_losses[it] = train_loss

    model.eval()
    test_loss = []
    for inputs, targets in iter(test_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss.append(loss.item())

    test_loss = np.mean(test_loss)
    test_losses[it] = test_loss

    # Save losses
    dt = datetime.now() - t0
    print(f'Epoch {it + 1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

model.eval()
sample = dataset[50]
output = model.forward(sample[0])
print(sample)
print(output)

print('\n------------------------test-----------------------------')
print('predicted', output.detach().numpy() * 90)
print('actual', sample[1].detach().numpy() * 90)
print('------------------------test-----------------------------\n')
print(dataset.means, dataset.stds, dataset.maxs)

x = np.array([[0, 65, 0]])
x = normalize_max(x, dataset.maxs)

input = torch.from_numpy(x.astype(np.float32))
output = model.forward(input).detach().numpy() * 90

print('\n------------------------test-----------------------------')
print(input)
print(np.floor(output))
print('------------------------test-----------------------------\n')
model.save_state()
