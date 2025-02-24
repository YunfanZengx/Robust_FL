import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate synthetic dataset
def generate_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    class_0 = np.random.normal(1, 1, (num_samples // 2, 2))  # Mean=1, Std=1
    class_1 = np.random.normal(4, 1, (num_samples // 2, 2))  # Mean=4, Std=1
    labels_0 = np.zeros(num_samples // 2)
    labels_1 = np.ones(num_samples // 2)
    X = np.vstack((class_0, class_1))
    y = np.hstack((labels_0, labels_1))
    return train_test_split(X, y, test_size=0.2, random_state=seed)

# Define Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Training and evaluation with mini-batches
def train_model_with_minibatch(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=50, batch_size=32):
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for mini-batches
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = LogisticRegressionModel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()  # Ensure model is in training mode
        for X_batch, y_batch in train_loader:
            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print loss at the end of each epoch
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate model
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        y_test_pred_class = (y_test_pred >= 0.5).float()
        accuracy = (y_test_pred_class.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
        print(f"Test Accuracy: {accuracy:.4f}")

    return model

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
        predictions = model(grid_tensor)
        predictions = (predictions >= 0.5).numpy().reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, predictions, alpha=0.6, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap=plt.cm.coolwarm)
    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

# Main program
if __name__ == "__main__":
    # Generate and split data
    X_train, X_test, y_train, y_test = generate_data(num_samples=1000)

    # Train logistic regression model with mini-batch SGD
    model = train_model_with_minibatch(X_train, y_train, X_test, y_test, learning_rate=0.1, epochs=50, batch_size=32)

    # Plot decision boundary
    plot_decision_boundary(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)))
