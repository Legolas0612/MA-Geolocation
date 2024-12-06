import os
from tkinter import Tk, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import transforms
import folium
import webview

# PyTorch Models
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

class GeoLocatorApp:
    def __init__(self, root, combined_model, simple_model, transform, device):
        self.root = root
        self.combined_model = combined_model
        self.simple_model = simple_model
        self.transform = transform
        self.device = device

        self.root.title("Image Geolocation Finder")
        self.root.geometry("800x400")

        # Layout
        self.image_label = Label(self.root, text="No Image Selected", bg="lightgray", width=30, height=10)
        self.image_label.grid(row=0, column=0, padx=10, pady=10)

        self.output_label = Label(self.root, text="", font=("Arial", 20))
        self.output_label.grid(row=1, column=0, padx=10, pady=10)

        self.choose_button = Button(self.root, text="Choose Image", command=self.choose_image)
        self.choose_button.grid(row=2, column=0, padx=10, pady=10)

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.locate_coordinates(file_path)

    def locate_coordinates(self, image_path):
        try:
            # Load and preprocess the image
            image = Image.open(image_path).convert('RGB')
            img_display = ImageTk.PhotoImage(image.resize((300, 300), Image.LANCZOS))
            self.image_label.config(image=img_display, width=300, height=300)
            self.image_label.image = img_display

            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Extract features using SimpleCNN
            self.simple_model.eval()
            with torch.no_grad():
                features = self.simple_model(input_tensor)

            # Predict coordinates using CombinedGeolocationCNN
            self.combined_model.eval()
            with torch.no_grad():
                predicted_coords = self.combined_model(input_tensor, features).squeeze().cpu().numpy()

            longitude, latitude = predicted_coords
            self.output_label.config(
                text=f"Predicted Coordinates:\nLatitude: {latitude:.6f}, Longitude: {longitude:.6f}"
            )

            # Update the Tkinter window to display the changes
            self.root.update()

            # Generate and display the map in a webview window
            map_dir = os.path.join(os.getcwd(), "temp_map.html")
            map_object = folium.Map(location=[latitude, longitude], zoom_start=6)
            folium.Marker([latitude, longitude], popup="Predicted Location").add_to(map_object)
            map_object.save(map_dir)

            # Open the map in a webview window
            webview.create_window("Predicted Location Map", map_dir)
            webview.start()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process the image: {e}")


def load_models(simple_model_path, combined_model_path, image_size, feature_size, device):
    simple_model = SimpleCNN(num_outputs=feature_size, image_size=image_size)
    simple_model.load_state_dict(torch.load(simple_model_path, map_location=device))
    simple_model.to(device)

    combined_model = CombinedGeolocationCNN(
        image_size=image_size, feature_size=feature_size,
        hidden_size1=128, hidden_size2=64, num_outputs=2
    )
    combined_model.load_state_dict(torch.load(combined_model_path, map_location=device))
    combined_model.to(device)

    return simple_model, combined_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = (512, 256)
    feature_size = 2259
    simple_model_path = input("Enter the path to the Climatezone model: ")
    #simple_model_path = "ClimateZone_epoch_20.pth"
    combined_model_path = input("Enter the path to the CombinedGeolocation model: ")
    #combined_model_path = "CombinedGeolocationCNN_20_epoch_20.pth"

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    simple_model, combined_model = load_models(
        simple_model_path, combined_model_path, image_size, feature_size, device
    )

    root = Tk()
    app = GeoLocatorApp(root, combined_model, simple_model, transform, device)
    root.mainloop()

if __name__ == '__main__':
    main()
