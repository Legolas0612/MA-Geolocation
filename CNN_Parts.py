import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

csv_file = 'parts.csv'
df = pd.read_csv(csv_file)

part_keys = [col for col in df.columns if col.startswith('part_')]

parts_data = {key: [] for key in part_keys}
for index, row in df.iterrows():
    image_name = row['name']
    image_path = os.path.join('Datasets/geolocation/images', image_name + '.jpeg')
    for part_key in part_keys:
        if row[part_key] != 0:
            parts_data[part_key].append(image_path)

train_data = {}
val_data = {}
for part_key, image_paths in parts_data.items():
    num_samples = len(image_paths)
    if num_samples == 0:
        #print(f"No samples for {part_key}. Skipping.")
        continue

    test_size = min(0.1, max(0.1, (num_samples - 1) / num_samples))
    train_paths, val_paths = train_test_split(image_paths, test_size=test_size, random_state=42)
    
    if len(train_paths) == 0 or len(val_paths) == 0:
        #print(f"Not enough samples for {part_key}. Skipping.")
        continue

    train_data[part_key] = train_paths
    val_data[part_key] = val_paths

transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LAT_DIVISIONS = 10
LON_DIVISIONS = 10

def calculate_bounds():
    lat_step = 180 / LAT_DIVISIONS
    lon_step = 360 / LON_DIVISIONS

    part_bounds = {}

    for i in range(LAT_DIVISIONS):
        lat_min = -90 + i * lat_step
        lat_max = lat_min + lat_step

        for j in range(LON_DIVISIONS):
            lon_min = -180 + j * lon_step
            lon_max = lon_min + lon_step

            part_key = f'part_{i}_{j}'
            part_bounds[part_key] = {
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max
            }

    return part_bounds

part_bounds = calculate_bounds()

class CustomDataset(Dataset):
    def __init__(self, image_paths, df, part_key, transform=None):
        self.image_paths = image_paths
        self.df = df
        self.part_key = part_key
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path).replace('.jpeg', '')
        target_row = self.df[self.df['name'] == image_name].iloc[0]
        
        image = Image.open(image_path).convert('RGB')
        target = torch.tensor([target_row['latitude'], target_row['longitude']], dtype=torch.float32)

        
        if self.transform:
            image = self.transform(image)
        
        return image, target 
    

class SimpleCNN(nn.Module):
    def __init__(self, lat_min, lat_max, lon_min, lon_max):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 64, 512)
        self.fc2 = nn.Linear(512, 2)
        
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = self.pool(nn.ReLU()(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 64)
        x = nn.ReLU()(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        lat_output = x[:, 0] * (self.lat_max - self.lat_min) / 2 + (self.lat_max + self.lat_min) / 2
        lon_output = x[:, 1] * (self.lon_max - self.lon_min) / 2 + (self.lon_max + self.lon_min) / 2
        return torch.stack([lat_output, lon_output], dim=1)

    def normalize_targets(self, lat_target, lon_target):
        lat_norm = 2 * (lat_target - self.lat_min) / (self.lat_max - self.lat_min) - 1
        lon_norm = 2 * (lon_target - self.lon_min) / (self.lon_max - self.lon_min) - 1
        return lat_norm, lon_norm




def train_model(part_key, train_dataset, epochs=5):
    AIs_Directory = 'Parts_AIs'
    bounds = part_bounds[part_key]
    model = SimpleCNN(
        lat_min=bounds['lat_min'], lat_max=bounds['lat_max'], 
        lon_min=bounds['lon_min'], lon_max=bounds['lon_max']
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            lat_target, lon_target = targets[:, 0], targets[:, 1]
            lat_norm, lon_norm = model.normalize_targets(lat_target, lon_target)
            normalized_targets = torch.stack([lat_norm, lon_norm], dim=1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, normalized_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss}')

    os.makedirs(AIs_Directory, exist_ok=True)
    torch.save(model.state_dict(), f'{AIs_Directory}/{part_key}_model.pth')



def test_model(part_key, val_dataset):
    differences = []
    AIs_Directory = 'Parts_AIs'
    model = SimpleCNN(
        lat_min=part_bounds[part_key]['lat_min'], lat_max=part_bounds[part_key]['lat_max'],
        lon_min=part_bounds[part_key]['lon_min'], lon_max=part_bounds[part_key]['lon_max']
    )
    model.load_state_dict(torch.load(f'{AIs_Directory}/{part_key}_model.pth'))
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            predictions = outputs.numpy()
            actuals = targets.numpy()

            diff = np.abs(predictions - actuals)
            differences.extend(diff)

    pd.DataFrame(differences, columns=['latitude_diff', 'longitude_diff']).to_csv(f'Distances_parts_AIs/{part_key}_differences.csv', index=False)

def main():

    for part_key in parts_data.keys():
        if part_key in train_data and part_key in val_data:
            train_dataset = CustomDataset(train_data[part_key], df, part_key, transform=transform)
            val_dataset = CustomDataset(val_data[part_key], df, part_key, transform=transform)
            train_model(part_key, train_dataset)
            test_model(part_key, val_dataset)
        else:
            print(f"Skipping {part_key} due to insufficient data.")

    print("Training complete!")


if __name__ == '__main__':
    main()