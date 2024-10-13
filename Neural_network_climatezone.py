import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

class GeoLocationDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0] + '.jpeg')
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        output = torch.tensor(self.data.iloc[idx, 3:].astype(float).values, dtype=torch.float32)
        return image, output

class SimpleCNN(nn.Module):
    def __init__(self, num_outputs, image_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

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
        x = self.sigmoid(x)
        return x

def main():
    print("libraries imported")
    image_folder = 'Datasets/geolocation/images'
    csv_file = 'distance_to_point_train.csv'
    learning_rate = 0.001
    num_epochs = 20
    image_size = (256, 512)
    batch_size = 32

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GeoLocationDataset(csv_file=csv_file, image_folder=image_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)

    print("Data loaded")
    num_outputs = 2259

    model = SimpleCNN(num_outputs, image_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("Starting training")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, outputs) in enumerate(dataloader):
            optimizer.zero_grad()

            outputs_pred = model(images)
            loss = criterion(outputs_pred, outputs)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / (i+1):.4f}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')
        torch.save(model.state_dict(), f'ClimateZone_epoch_{epoch+1}.pth')


if __name__ == '__main__':
    main()
    print('Training Finished')