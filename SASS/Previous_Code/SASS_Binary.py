import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset


# ------------------------------
# 1. Dataset: Gaussian Binary Classification
# ------------------------------
class GaussianBinaryDataset(Dataset):
    def __init__(self, n_samples=1000, dim=2):
        """
        Synthetic binary classification dataset:
        - Class 0: Gaussian(mean=1, std=1).
        - Class 1: Gaussian(mean=3, std=1).
        """
        np.random.seed(42)

        # Number of samples per class
        n_class = n_samples // 2

        # Class 0 (label 0)
        class_0 = np.random.normal(loc=1, scale=1, size=(n_class, dim))

        # Class 1 (label 1)
        class_1 = np.random.normal(loc=3, scale=1, size=(n_class, dim))

        # Combine and assign labels
        self.data = np.vstack([class_0, class_1])
        self.labels = np.array([0] * n_class + [1] * n_class)

        # Convert to PyTorch tensors
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ------------------------------
# 2. Model Class: Binary Classifier
# ------------------------------
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, 2)

    def forward(self, x):
        return self.fc(x)


# ------------------------------
# 3. SASS Training Loop
# ------------------------------
def train_sass(model, train_loader, test_loader, epochs, step_size_init, theta, gamma, eps_f):
    """
    Trains the model using the SASS algorithm.

    Args:
        model: PyTorch model to train.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
        epochs: Number of epochs.
        step_size_init: Initial step size.
        theta, gamma, eps_f: SASS hyperparameters.

    Returns:
        Accuracy, learning rates, and gradient norms.
    """
    # Initialize optimizer state
    step_size = step_size_init
    step_sizes = []  # Track learning rates
    grad_norms = []  # Track gradient norms
    accuracies = []  # Track test accuracies

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            # Forward pass
            output = model(data)
            loss = loss_fn(output, target)

            # Compute gradients
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
            grad_norms.append(grad_norm.item())

            # Propose step
            with torch.no_grad():
                for param, grad in zip(model.parameters(), grads):
                    param -= step_size * grad

            # Check Armijo condition
            output_new = model(data)
            loss_new = loss_fn(output_new, target)

            armijo_condition = loss_new <= loss - step_size * theta * (grad_norm ** 2) + 2 * eps_f
            if armijo_condition:
                step_size = min(step_size / gamma, 1.0)  # Increase step size
            else:
                step_size = max(step_size * gamma, 1e-6)  # Decrease step size

            step_sizes.append(step_size)

        # Evaluate model on test data
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accuracy = correct / total * 100
        accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}: Test Accuracy = {accuracy:.2f}%, Step Size = {step_size:.6f}")

    return accuracies, step_sizes, grad_norms


# ------------------------------
# 4. Main Function
# ------------------------------
def main():
    # Parameters
    n_samples = 1000
    dim = 2
    batch_size = 32
    epochs = 20
    step_size_init = 0.1
    theta = 0.1
    gamma = 1.1
    eps_f = 1e-4

    # Prepare dataset
    dataset = GaussianBinaryDataset(n_samples=n_samples, dim=dim)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = BinaryClassifier(input_dim=dim)

    # Train model using SASS
    accuracies, step_sizes, grad_norms = train_sass(
        model, train_loader, test_loader, epochs, step_size_init, theta, gamma, eps_f
    )

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o')
    plt.title("Test Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(step_sizes) + 1), step_sizes, marker='o')
    plt.title("Learning Rate Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Step Size")
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(grad_norms) + 1), grad_norms, marker='o')
    plt.title("Gradient Norm Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
