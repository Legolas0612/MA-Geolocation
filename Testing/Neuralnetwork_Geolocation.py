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

input_size = 319488

model = Neuralnetwork(input_size, hidden_size1=128, hidden_size2=64, output_size=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1

targets = []
for line in open('train_coordinates.csv', 'r'):
    targets.append(line.strip().split(','))
targets = [[float(target[0]), float(target[1])] for target in targets]

print(len(targets))

for epoch in range(num_epochs):
    with open('train.csv', 'r') as file1:
        for j, line in enumerate(file1, start=0):
            inputs = []
            line = line.strip().strip('[]')
            values = line.split(',')
            for value in values:
                grayscale_value = float(value.strip().strip("'"))
                inputs.append(float(grayscale_value))
            if len(inputs) >= 319488:
                inputs = inputs[:319488]
            elif len(inputs) < 319488:
                while len(inputs) < 319488:
                    inputs.append(0)
            inputs = np.array(inputs, dtype=np.float32) 
            inputs = torch.from_numpy(inputs)
            target = np.array(targets[j], dtype=np.float32) 
            target = torch.from_numpy(target)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            if j % 100 == 0:
                print(j)
            if j == len(targets) - 1:
                break
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), '100000_1_128_64.pth')