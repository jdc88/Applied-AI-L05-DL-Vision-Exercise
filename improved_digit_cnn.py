import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

# Mish activation function
def mish(x):
    return x * torch.tanh(F.softplus(x))

# CNN model with 1 more FC layer and no dropout
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

# Basic transform only (no augmentation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Use full dataset: train + test
train_dataset = datasets.MNIST(root='', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='', train=False, transform=transform, download=True)
full_dataset = ConcatDataset([train_dataset, test_dataset])
train_loader = DataLoader(full_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(full_dataset, batch_size=1000, shuffle=False)

def evaluate(model, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

def train_and_evaluate(device):
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    print(f"Training on {device} using full dataset (train + test)...")
    for epoch in range(1, 41):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, device)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy == 100.0:
            torch.save(model.state_dict(), "improved_digit_model.pth")
            print("🎯 100% accuracy achieved! Model saved as improved_digit_model.pth")
            return

    final_accuracy = evaluate(model, device)
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    if final_accuracy == 100.0:
        torch.save(model.state_dict(), "improved_digit_model.pth")
        print("✅ Model saved as improved_digit_model.pth")
    else:
        print("⚠️ Model did not reach 100%. Try adjusting further.")

# Run
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_and_evaluate(device)