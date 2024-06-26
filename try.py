import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Neuralnetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, hidden_size2):
        super(Neuralnetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x

input_size = 393216

model = Neuralnetwork(input_size, hidden_size1=128, hidden_size2=64, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 2

for epoch in range(num_epochs):
    with open('train.txt', 'r') as file1, open('train_coordinates.txt', 'r') as file2:
        for i in range(1, 100000):
            print(i)
            for j, line in enumerate(file1, start=i):
                if j == i + 1:
                    inputs = line.strip().split(',')
                    inputs = torch.tensor([int(x) for x in inputs])
                    break
            for j, line in enumerate(file2, start=i):
                if j == i + 1:
                    targets = line.strip().split(',')
                    targets = torch.tensor([float(x) for x in targets])
                    break
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'geolocation.pth')