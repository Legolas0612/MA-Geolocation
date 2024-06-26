import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import islice

class Neuralnetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
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
hidden_size1 = 128
hidden_size2 = 64
output_size = 2

model = Neuralnetwork(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5

def get_line(file, line_number):
    with open(file, 'r') as f:
        line = next(islice(f, line_number, line_number+1), None)
        if line is not None:
            return np.array(line.strip().split(','), dtype=np.uint8)
        return None
    
def get_inputs_as_npArray(file):
    inputs = []
    i = 0
    print("test2")
    with open(file, 'r') as f:
        for line in f:
            a = np.array(line.strip().split(','), dtype=np.uint8)
            if len(line) != 319488:
                #reduce length of line
                a = a[:319488]
            inputs.append(a)
            print(i)
            i += 1
    return np.array(inputs)

def get_targets_as_npArray(file):
    targets = []
    print("test3")
    i = 0
    with open(file, 'r') as f:
        for line in f:
            print(i)
            a = np.array(line.strip().split(','), dtype=np.float32)
            targets.append(a)
            i += 1


for epoch in range(num_epochs):
    print("test1")
    inputs = get_inputs_as_npArray('train.txt')
    targets = get_targets_as_npArray('train_coordinates.txt')
    print("test4")
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    for i in range(len(inputs)):
        inputs = inputs[i]
        targets = targets[i]
        optimizer.zero_grad()

        outputs = model(inputs.float())

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()
        print(i)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'geolocation5.pth')
