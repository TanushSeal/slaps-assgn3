import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# ==========================================
# PART 1.1: DEFINITIONS (Classes & Setup)
# ==========================================

# 1. Define the Dataset Wrapper
class MNISTSumDataset(Dataset):
    def __init__(self, mnist_data):
        self.mnist_data = mnist_data
    
    def __len__(self):
        return len(self.mnist_data) // 2
    
    def __getitem__(self, index):
        img1, label1 = self.mnist_data[2*index]
        img2, label2 = self.mnist_data[2*index+1]
        return (img1, img2), label1 + label2

# 2. Define the Neural Network
class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 128)
        self.fc_final = nn.Linear(128 * 2, 19) # 19 classes (sums 0-18)

    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_one(img1)
        feat2 = self.forward_one(img2)
        combined = torch.cat((feat1, feat2), dim=1)
        return self.fc_final(combined)

# ==========================================
# PART 1.2: EXECUTION (Train & Test)
# ==========================================

def main():
    # 1. Setup Device and Data
    print("Preparing Data...")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(MNISTSumDataset(mnist_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(MNISTSumDataset(mnist_test), batch_size=64, shuffle=False)

    # 2. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaselineCNN().to(device)
    
    # IMPROVEMENT: Use a scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) # Decays LR every 5 epochs
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {device}...")

    # 3. Extended Training Loop
    epochs = 50  # Increased from 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, ((img1, img2), target) in enumerate(train_loader):
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f} (LR: {current_lr:.1e})")

    # 4. Testing Loop
    print("Evaluating...")
    model.eval()
    correct = 0
    with torch.no_grad():
        for (img1, img2), target in test_loader:
            img1, img2, target = img1.to(device), img2.to(device), target.to(device)
            output = model(img1, img2)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            
    print(f'Final Baseline Accuracy: {100. * correct / len(test_loader.dataset):.2f}%')


    print("Saving Baseline Model to 'baseline_cnn.pth'...")
    torch.save(model.state_dict(), "baseline_cnn.pth")
    print("Saved successfully!")

if __name__ == '__main__':
    main()