import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 📁 Dataset folder
data_dir = "L05_DL_Vision_Dogs"

# 🐕 Known breed names (labels)
breed_names = ["Beagle", "Boxer", "Bulldog", "Golden Retriever", "Labrador Retriever", "Poodle"]
breed_to_label = {breed: i for i, breed in enumerate(breed_names)}

# 📦 Custom Dataset class
class DogDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.samples = []

        for filename in os.listdir(folder):
            if filename.lower().endswith(".jpg"):
                for breed in breed_names:
                    if filename.startswith(breed):
                        self.samples.append((os.path.join(folder, filename), breed_to_label[breed]))
                        break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# 🧪 Data Augmentation and Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Data augmentation
    transforms.RandomRotation(20),      # Random rotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 🧹 Load data and split
dataset = DogDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

# 🧠 Load pretrained ResNet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(breed_names))
model = model.to(device)

# Fine-tuning: Freeze all layers except the final layer
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers

# Unfreeze the final fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True  # Make sure the final layer is trainable

# 📊 Print architecture summary
print("📚 Model Summary (ResNet18):")
print(model)
print(f"🧠 Layers: {sum(1 for _ in model.modules())}")
print(f"🎛️  Total Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"✅ Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}\n")

# 🎯 Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Use a lower learning rate

# 🔁 Training with learning rate scheduler
num_epochs = 10  # Increase the number of epochs for better training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)  # Learning rate scheduler

print("⏳ Training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    scheduler.step()  # Adjust the learning rate based on the scheduler
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 🧪 Evaluation
print("\n🔍 Evaluating...")
model.eval()
correct = 0
total = 0
start_time = time.time()

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

elapsed_time = time.time() - start_time
accuracy = 100 * correct / total

print(f"\n📈 Accuracy: {accuracy:.2f}%")
print(f"⏱️  Classification Time: {elapsed_time:.2f} seconds")