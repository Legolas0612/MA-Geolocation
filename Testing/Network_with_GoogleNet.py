import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
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

class CombinedGeolocationCNN(nn.Module):
    def __init__(self, feature_size, hidden_size1, hidden_size2, num_outputs):
        super(CombinedGeolocationCNN, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, num_outputs)

    def forward(self, features):
        x = self.fc1(features)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def main():
    print("Libraries imported")
    image_folder = 'Datasets/geolocation/images'
    csv_file = 'Datasets/geolocation/images.csv'
    learning_rate = 0.001
    num_epochs = 20
    image_size = (224, 224)
    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    googlenet = models.googlenet(pretrained=True)
    googlenet.fc = nn.Identity()
    googlenet.eval()

    dataset = GeoLocationDataset(
        csv_file=csv_file,
        image_folder=image_folder,
        transform=transform,
        feature_extractor=googlenet
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    print("Data loaded")

    feature_size = 1024
    num_outputs = 2
    combined_model = CombinedGeolocationCNN(feature_size=feature_size, hidden_size1=512, hidden_size2=256, num_outputs=num_outputs)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)

    print("Starting training")
    for epoch in range(num_epochs):
        combined_model.train()
        running_loss = 0.0
        for i, (_, features, outputs) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs_pred = combined_model(features)
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
