import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class GeoLocationTestDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, feature_extractor=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data[self.data.index >= 150000])

    def __getitem__(self, idx):
        idx += 150000
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0] + '.jpeg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)

        with torch.no_grad():
            features = self.feature_extractor(image.unsqueeze(0)).squeeze()

        latitude = self.data.at[idx, 'latitude']
        longitude = self.data.at[idx, 'longitude']

        output = torch.tensor([latitude, longitude], dtype=torch.float32)
        return image, features, output

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs, image_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

        flattened_size = 64 * (image_size[0] // 8) * (image_size[1] // 8)
        self.fc = nn.Linear(flattened_size, num_outputs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CombinedGeolocationCNN(nn.Module):
    def __init__(self, image_size, feature_size, hidden_size1, hidden_size2, num_outputs):
        super(CombinedGeolocationCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

        flattened_size = 128 * (image_size[0] // 16) * (image_size[1] // 16) + feature_size

        self.fc1 = nn.Linear(flattened_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_outputs)

    def forward(self, image, features):
        x = self.conv1(image)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, features), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def test_model(combined_model, test_loader, device):
    combined_model.eval()
    difference_latitudes = []
    difference_longitudes = []
    latitude_targets = []
    longitude_targets = []
    latitude_predictions = []
    longitude_predictions = []
    image_names = []

    with torch.no_grad():
        for images, features, targets in test_loader:
            images, features, targets = images.to(device), features.to(device), targets.to(device)

            outputs = combined_model(images, features)
            difference = torch.abs(outputs - targets)
            latitude_targets.extend(targets[:, 0].tolist())
            longitude_targets.extend(targets[:, 1].tolist())
            latitude_predictions.extend(outputs[:, 0].tolist())
            longitude_predictions.extend(outputs[:, 1].tolist())
            difference_latitudes.extend(difference[:, 0].tolist())
            difference_longitudes.extend(difference[:, 1].tolist())
            
            print(f"Latitude difference: {difference[0][0]:.6f}, Longitude difference: {difference[0][1]:.6f}")

    mean_latitude = sum(difference_latitudes) / len(difference_latitudes)
    mean_longitude = sum(difference_longitudes) / len(difference_longitudes)
    
    print(f'Mean Latitude Difference: {mean_latitude:.6f}')
    print(f'Mean Longitude Difference: {mean_longitude:.6f}')
    image_names = pd.read_csv('Datasets/geolocation/images.csv').iloc[150000:]['name'].tolist()
    pd.DataFrame({'image_name': image_names, 'latitude_difference': difference_latitudes, 'longitude_difference': difference_longitudes, "latitude_targets": latitude_targets, 'longitude_targets': longitude_targets, 'latitude_predictions': latitude_predictions, "longitude_predictions": longitude_predictions}).to_csv('results.csv', index=False)


def main():
    image_folder = 'Datasets/geolocation/images'
    csv_file = 'Datasets/geolocation/images.csv'
    pretrained_model_path = 'Climate_zone_epoch_10.pth'
    combined_model_path = 'CombinedGeolocationCNN_epoch_10.pth'
    image_size = (224, 224)
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_size = 2259
    pretrained_model = SimpleCNN(num_outputs=feature_size, image_size=image_size)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path))
    pretrained_model.eval()

    test_dataset = GeoLocationTestDataset(
        csv_file=csv_file,
        image_folder=image_folder,
        transform=transform,
        feature_extractor=pretrained_model
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True)

    combined_model = CombinedGeolocationCNN(image_size=image_size, feature_size=feature_size, hidden_size1=128, hidden_size2=64, num_outputs=2)
    combined_model.load_state_dict(torch.load(combined_model_path))
    combined_model.to(device)

    test_model(combined_model, test_loader, device)

if __name__ == '__main__':
    main()
