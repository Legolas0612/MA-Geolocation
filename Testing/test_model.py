import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

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


model_path = "100000_1_128_64.pth"
model.load_state_dict(torch.load(model_path))


model.eval()

with open('test_coordinates.csv', 'r') as file1:
    outputs = []
    for line in file1:
        line = line.strip().split(',')
        outputs.append([float(line[0]), float(line[1])])
    outputs = np.array(outputs, dtype=np.float32)

X_test = []
outputs = torch.tensor(outputs)

X_test_tensors = []
with torch.no_grad():
    with open('rgb_values4.txt', 'r') as file1:
        for j, line in enumerate(file1, start=0):
            if j < 99999:
                continue
            else:
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
                inputs = torch.from_numpy(inputs).unsqueeze(0) 
                X_test_tensors.append(model(inputs))
            if j % 100 == 0:
                print(j)
            if j == len(outputs) - 1:
                break

    X_test_stacked = torch.vstack(X_test_tensors)

    mse = torch.mean((outputs - X_test_stacked)**2)
    print(f'Mean squared error: {mse}')
    
    mae = torch.mean(torch.abs(outputs - X_test_stacked))
    print(f'Mean absolute error: {mae}')