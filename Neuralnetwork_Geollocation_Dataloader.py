import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class GeoLocationDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None, feature_extractor=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= 150000:
            idx = 0
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

def main():
    print("Libraries imported")
    image_folder = 'Datasets/geolocation/images'
    csv_file = 'Datasets/geolocation/images.csv'
    pretrained_model_path = 'ClimateZone_epoch_20.pth'
    learning_rate = 0.001
    num_epochs = 20
    image_size = (512, 256)
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    feature_size = 2259
    pretrained_model = SimpleCNN(num_outputs=feature_size, image_size=image_size)
    pretrained_model.load_state_dict(torch.load(pretrained_model_path))
    pretrained_model.eval()

    dataset = GeoLocationDataset(
        csv_file=csv_file,
        image_folder=image_folder,
        transform=transform,
        feature_extractor=pretrained_model
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    print("Data loaded")

    num_outputs = 2
    combined_model = CombinedGeolocationCNN(image_size=image_size, feature_size=feature_size, hidden_size1=128, hidden_size2=64, num_outputs=num_outputs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    print("Starting training")
    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0
        for i, (images, features, outputs) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs_pred = combined_model(images, features)
            loss = criterion(outputs_pred, outputs)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / (i+1):.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
        torch.save(combined_model.state_dict(), f'CombinedGeolocationCNN_epoch_{epoch+1}.pth')

    print('Training Finished')

if __name__ == '__main__':
    main()
