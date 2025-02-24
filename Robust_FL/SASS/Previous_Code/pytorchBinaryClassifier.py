import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Data Generation
# Class 0: Gaussian(0, 1)
class_0 = np.random.normal(loc=0, scale=1, size=(1000, 2))

# Class 1: Gaussian(3, 1)
class_1 = np.random.normal(loc=3, scale=1, size=(1000, 2))

# Labels
labels_0 = np.zeros((1000, 1))  # Label 0 for class 0
labels_1 = np.ones((1000, 1))   # Label 1 for class 1

# Combine data and labels
X = np.vstack((class_0, class_1))  # Combine features
y = np.vstack((labels_0, labels_1))  # Combine labels

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 2. Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear layer

    def forward(self, x):
        return torch.sigmoid(self.linear(x))  # Sigmoid activation

# Model instance
input_dim = X_train.shape[1]  # Number of features
model = LogisticRegressionModel(input_dim)

# 3. Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# 4. Training Loop
num_epochs = 200
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    # Binary Cross Entropy Loss
    loss = nn.BCELoss()(outputs, y_train_tensor) 

    # Backward pass
    optimizer.zero_grad() # Clears any previously computed gradients for (w,b),ensures that updates are based only current batch.
    loss.backward() # Computes the gradient of the loss w.r.t model parameter (w,b)
    optimizer.step() #update model parameter : w = w - lr * gradient(w); b = ...


# 5. Evaluate the Model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation for efficiency
    y_pred = model(X_test_tensor)
    y_pred_labels = (y_pred >= 0.5).float()  # Convert probabilities to binary labels
    accuracy = (y_pred_labels.eq(y_test_tensor).sum() / y_test_tensor.shape[0]).item()
    print(f'Accuracy on test set: {accuracy:.4f}')

# 6. Visualization of Data and Decision Boundary
def plot_decision_boundary(X, y, model):
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    
    # Predict probabilities for the grid
    with torch.no_grad():
        probs = model(grid_tensor).reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.7, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', cmap='bwr')
    plt.title('Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Combine test and train for visualization
X_combined = np.vstack((X_train, X_test))
y_combined = np.vstack((y_train, y_test))
plot_decision_boundary(X_combined, y_combined, model)
