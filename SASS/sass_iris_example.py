import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sass import Sass
import time

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Define a simple neural network
class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Initialize the model and optimizer
model = IrisNet()
n_batches_per_epoch = len(train_loader)
optimizer = Sass(model.parameters(), n_batches_per_epoch=n_batches_per_epoch)

# Training loop
num_epochs = 100
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss

        loss = optimizer.step(closure)

    # Evaluation
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(test_loader.dataset)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    if time.time() - start_time > 60:  # Stop if training takes more than 1 minute
        print("Training stopped due to time limit.")
        break

end_time = time.time()
print(f'Training completed in {end_time - start_time:.2f} seconds.')
print(f'Final Test Accuracy: {accuracy:.2f}%')