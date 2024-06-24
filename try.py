import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np


class Neuralnetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, hidden_size2):
        super(model, self).__init__()
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

class ImageDataset(Dataset):
    def __init__ (self, file_paths):
        self.file_paths = file_paths
        self.data_indices = self._prepare_data_indices()

    def _prepare_data_indices(self):
        data_indices = []
        for file_path in self.file_paths:
            with open(file_path, 'r') as f:
                offset = 0
                for line in f:
                    length = len(line)
                    data_indices.append((file_path, offset, length, line))
                    offset += len(line)
        return data_indices
    
    def __len__(self):
        return len(self.data_indices)
    
    def __getitem__(self, idx):
        file_path, offset, length, = self.data_indices[idx]
        with open(file_path, 'r') as f:
            f.seek(offset)
            data = f.read(length)
            data = data.strip().split(',')
            inputs = torch.tensor([int(x) for x in data])
            with open('coordinates.txt', 'r') as file:
                for i, line in enumerate(file, start=1):
                    if i == idx + 1:
                        outputs = line.strip().split(',')
                        outputs = torch.tensor([float(x) for x in outputs])
            return inputs, outputs

        
file_paths = ['rgb_values.txt']

dataset = ImageDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Neuralnetwork(input_size=784, hidden_size1=128, hidden_size2=64, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()