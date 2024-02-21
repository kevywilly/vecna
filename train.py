import torch
from datetime import datetime
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from src.kinematics import Dataset
import os


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

model = nn.Sequential(
    nn.Linear(3,16),
    nn.Linear(16,32),
    nn.Linear(32,64),
    nn.ReLU(),
    nn.Linear(64,3)
)

best_model = '/Users/kevywilly/Projects/vecna/best_model.pth'

if os.path.isfile(best_model):
    model.load_state_dict(torch.load(best_model))

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
    print(f'Epoch {it+1}/{epochs}, Train Loss: {train_loss:.4f}, \
      Test Loss: {test_loss:.4f}, Duration: {dt}')


    if test_loss < best_loss:
        #torch.save(model.state_dict(), best_model)
        best_loss = test_loss

plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

model.eval()
sample = dataset[50]
output = model.forward(sample[0])
print(sample)
print(output)
print('predicted', output.detach().numpy()*90)
print('actual', sample[1].detach().numpy()*90)

print(dataset.means, dataset.stds, dataset.maxs)

means = dataset.means
stds = dataset.stds
maxs = dataset.maxs
x = np.array([[0,65,0]])
x = Dataset.normalize(x,means,stds,maxs)
print(x)
input = torch.from_numpy(x.astype(np.float32))
output = model.forward(input).detach().numpy()*90
print(input)
print(np.floor(output))