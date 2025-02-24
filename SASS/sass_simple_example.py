import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sass import Sass  # Assuming the SASS optimizer is in a file named sass.py

# Create a dataset with signal
def create_dataset(num_samples=1000, num_features=10):
    X = torch.randn(num_samples, num_features)
    y = (X[:, :5].sum(dim=1) > 0).float()
    return X, y

# Create train and test datasets
X_train, y_train = create_dataset(num_samples=1000)
X_test, y_test = create_dataset(num_samples=200)

# Create train DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model
model = SimpleNet()

# Initialize the SASS optimizer
optimizer = Sass(model.parameters(), n_batches_per_epoch=len(train_dataloader))

# Loss function
criterion = nn.BCELoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_dataloader:
        def closure():
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            return loss
        
        loss = optimizer.step(closure)
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

# Test the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = (test_outputs.squeeze() > 0.5).float()
    accuracy = (test_predictions == y_test).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    print("Sample predictions:", test_outputs[:5].squeeze())
    print("Actual labels:    ", y_test[:5])