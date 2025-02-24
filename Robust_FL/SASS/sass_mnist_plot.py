import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sass import Sass  # Assuming the SASS optimizer is in a file named sass.py
import os
import matplotlib.pyplot as plt

# Set up data directory
data_dir = './mnist_data'
os.makedirs(data_dir, exist_ok=True)
save_path = "./output_SASS_MNIST_plots"
os.makedirs(save_path, exist_ok=True)
# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset using torchvision
train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

# Create data loaders with smaller batch size
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

# Define a smaller neural network
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize the model and optimizer
model = SmallNet()
n_batches_per_epoch = len(train_loader)
optimizer = Sass(model.parameters(), n_batches_per_epoch=n_batches_per_epoch)

# Training loop
num_epochs = 50
train_losses = []
accuracy_lst = []
learning_rates = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    epoch_step_sizes = []
    
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss
        
        loss = optimizer.step(closure)
        epoch_loss += loss.item()
        epoch_step_sizes.append(optimizer.state['step_size'])

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]	Loss: {loss.item():.6f}')
    
    avg_step_size = np.mean(epoch_step_sizes)
    learning_rates.append(avg_step_size)
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    print(f'Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.6f}, Average Learning Rate: {avg_step_size:.6f}')

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    accuracy_lst.append(accuracy)


# Plot the training loss over epochs
loss_plot_path = os.path.join(save_path, "training_loss.png")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs for SASS")
plt.grid(True)
plt.savefig(loss_plot_path)
print(f"Training loss plot saved at: {loss_plot_path}")

# Plot the test accuracy over epochs
accuracy_plot_path = os.path.join(save_path, "test_accuracy.png")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), accuracy_lst, marker='o', linestyle='-', color='g')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy over Epochs for SASS")
plt.grid(True)
plt.savefig(accuracy_plot_path)
print(f"Test accuracy plot saved at: {accuracy_plot_path}")

# Plot the learning rate over epochs
learning_rate_plot_path = os.path.join(save_path, "learning_rate_per_epoch.png")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), learning_rates, marker='o', linestyle='-', color='orange')
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Epoch for SASS")
plt.grid(True)
plt.savefig(learning_rate_plot_path)
print(f"Learning rate plot saved at: {learning_rate_plot_path}")

print('Training completed.')
