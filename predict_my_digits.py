# Predict My Digits - Improved Version (Jupyter Notebook)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import os
from IPython.display import display

# Define the Mish activation function
def mish(x):
    return x * torch.tanh(F.softplus(x))

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = mish(self.conv1(x))
        x = self.pool1(mish(self.conv2(x)))
        x = mish(self.conv3(x))
        x = self.pool2(mish(self.conv4(x)))
        x = self.pool3(mish(self.conv5(x)))
        x = x.view(-1, 512 * 4 * 4)
        x = mish(self.fc1(x))
        x = mish(self.fc2(x))
        x = self.fc3(x)
        return torch.log_softmax(x, dim=1)

# Load model
model_path = "improved_digit_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("✅ Model loaded successfully.")

# Define transform (normalized for MNIST)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Define image preprocessing and prediction function
def preprocess_and_predict(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Load and convert to grayscale
    image = Image.open(image_path).convert("L")
    
    # Invert image if it's dark-on-light instead of light-on-dark
    image = ImageOps.invert(image)

    # Display original
    display(image)

    # Resize and apply transformation
    image_resized = image.resize((28, 28), Image.LANCZOS)
    image_tensor = transform(image_resized).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).item()

    print(f"✅ Prediction for {os.path.basename(image_path)}: {pred}")

# Predict on your test digit images
for filename in ["digit3.jpeg", "digit4.jpeg", "digit5.jpeg"]:
    preprocess_and_predict(filename)