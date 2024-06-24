import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Assuming the model definition remains the same

class CustomDataset(Dataset):
    def __init__(self, pixel_file, coord_file):
        # Load pixel values
        with open(pixel_file, 'r') as f:
            self.X = np.array([list(map(float, line.strip().split())) for line in f.readlines()], dtype=np.float32)
        
        # Load coordinates
        with open(coord_file, 'r') as f:
            self.y = np.array([list(map(float, line.strip().split())) for line in f.readlines()], dtype=np.float32)
        
        # Normalize pixel values to [0, 1]
        self.X /= 255.0

        # Convert arrays to tensors
        self.X = torch.tensor(self.X)
        self.y = torch.tensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Paths to your data files
pixel_file = 'rgb_values.txt'
coord_file = 'path/to/coordinates.txt'

# Initialize dataset
train_dataset = CustomDataset(pixel_file, coord_file)
# Assuming you have a way to split your dataset into train and test
# For simplicity, using the same dataset for both train and test here
test_dataset = CustomDataset(pixel_file, coord_file)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, criterion, and optimizer as before
# Assuming input_size, hidden_size1, hidden_size2, and output_size are defined
model = NeuralNetwork(input_size, hidden_size1, output_size, hidden_size2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 15
for epoch in range(epochs):
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets.float())  # Ensure targets are float type
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}')

# Save and evaluate the model as before