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
save_path="/home/local/ASURITE/yzeng88/fedSASS/SASS/output"


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
train_losses = [] # Reduced number of epochs
accuracy_lst =[]
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0  # To accumulate loss over the epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss

        loss = optimizer.step(closure)
        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
    # Calculate and store average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_epoch_loss)
    print(f'Epoch {epoch + 1} Average Training Loss: {avg_epoch_loss:.6f}')

    # Evaluation
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
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
# Plot the training loss over epochs
loss_plot_path = os.path.join(save_path, "training_loss.png")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss over Epochs for SASS")
plt.savefig(loss_plot_path)
plt.grid(True)


accuracy_plot_path = os.path.join(save_path, "test_accuracy.png")
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), accuracy_lst, marker='o', linestyle='-', color='g')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy over Epochs for SASS")
plt.grid(True)
plt.savefig(accuracy_plot_path)
print(f"Test accuracy plot saved at: {accuracy_plot_path}")

print('Training completed.')

# Plot Training Loss vs NN Passes
loss_vs_passes = optimizer.state['loss_vs_nn_passes']
# print(f"Number of entries in loss_vs_nn_passes: {len(loss_vs_passes)}")
# print(f"Sample entries in loss_vs_nn_passes: {loss_vs_passes[:5]}")  # Print first 5 entries
passes, losses = zip(*loss_vs_passes)  # Separate into x and y

# Plot and save the figure
nn_passes_plot_path = os.path.join(save_path, "training_loss_vs_nn_passes.png")
plt.figure(figsize=(8, 6))
plt.plot(passes, losses, marker='o', linestyle='-', color='r')
plt.xlabel("Number of NN Passes")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Number of NN Passes for SASS")
plt.grid(True)
plt.savefig(nn_passes_plot_path)
print(f"Training loss vs NN passes plot saved at: {nn_passes_plot_path}")

# Plot Step Size vs NN Passes
step_size_vs_passes = optimizer.state['step_size_vs_nn_passes']
if step_size_vs_passes:
    nn_passes, step_sizes = zip(*step_size_vs_passes)  # Separate into x and y

    # Plot and save the figure
    step_size_plot_path = os.path.join(save_path, "step_size_vs_nn_passes.png")
    plt.figure(figsize=(8, 6))
    plt.plot(nn_passes, step_sizes, marker='o', linestyle='-', color='purple')
    plt.xlabel("Number of NN Passes")
    plt.ylabel("Step Size")
    plt.title("Step Size vs Number of NN Passes for SASS")
    plt.grid(True)
    plt.savefig(step_size_plot_path)
    print(f"Step size vs NN passes plot saved at: {step_size_plot_path}")
else:
    print("No data available in step_size_vs_nn_passes for plotting.")
