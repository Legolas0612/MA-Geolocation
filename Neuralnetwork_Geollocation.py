import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import PIL as pil
from PIL import Image
from functions import get_rgb_values, get_imageId_latitude_longitude_values

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, hidden_size2=32):
        super(NeuralNetwork, self).__init__()
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

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the dataset
image_id, latitude, longitude = get_imageId_latitude_longitude_values()

# Get the RGB values for each image
rgb_values = []
a = 0
for id in image_id:
    rgb_values = get_rgb_values(str(id))
    with open('rgb_values.txt', 'a') as f:
    # Write each value on a new line
        for value in rgb_values:
            f.write(str(value))
        f.write('\n')
    a += 1
    with open('counter.txt', 'a') as f:
        f.write(str(a))
        f.write('\n')
        print(a)


# Convert latitude and longitude to tensors
latitude = torch.tensor(latitude.values).float()
longitude = torch.tensor(longitude.values).float()

# Concatenate latitude and longitude along the second dimension
y = torch.stack((latitude, longitude), dim=1)

# Split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(rgb_values, y, test_size=0.2, random_state=42)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

# Define hyperparameters
input_size = X_train.shape[1]
hidden_size = 64
output_size = 2 

# Instantiate the model
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Use mean squared error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 15
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    
    # Compute loss
    loss = criterion(outputs, y_train.float())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    print('Epoch', epoch+1 ,"/", epochs)

# Save the model after training
torch.save(model.state_dict(), 'image_geolocation.pth')

# Then you can load it later for evaluation
model.load_state_dict(torch.load('image_geolocation.pth'))

# Evaluation
model.eval()

with torch.no_grad():  # don't compute gradients
    outputs = model(X_test)
    print(f'Predicted values for test set: {outputs}')

    # Calculate the mean squared error between the predicted and actual values
    mse = torch.mean((outputs - y_test)**2)
    print(f'Mean squared error: {mse}')

    # Calculate the mean absolute error between the predicted and actual values
    mae = torch.mean(torch.abs(outputs - y_test))
    print(f'Mean absolute error: {mae}')