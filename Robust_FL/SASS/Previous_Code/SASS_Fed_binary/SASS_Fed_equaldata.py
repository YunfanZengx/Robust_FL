import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.utils import shuffle

EPS = 1e-6  # Numerical stability

# Logistic loss function
def f(xi, yi, opt_x):
    h_val = 1 / (1 + np.exp(-np.dot(xi, opt_x)))
    h_val = np.clip(h_val, EPS, 1 - EPS)
    return -yi * np.log(h_val) - (1 - yi) * np.log(1 - h_val)

# Gradient of logistic loss
def grad_f(xi, yi, opt_x):
    h_val = 1 / (1 + np.exp(-np.dot(xi, opt_x)))
    return -(yi - h_val) * xi

# Zeroth-order oracle: Compute noisy function value
def zeroth_oracle(f, x, y, opt_x, sample_size):
    sample_list = np.random.choice(range(len(x)), sample_size, replace=False)
    return np.mean([f(x[i], y[i], opt_x) for i in sample_list])

# First-order oracle: Compute gradient approximation
def first_oracle(grad_f, x, y, opt_x, sample_size):
    sample_list = np.random.choice(range(len(x)), sample_size, replace=False)
    grad = np.zeros_like(opt_x)
    for i in sample_list:
        grad += grad_f(x[i], y[i], opt_x)
    return grad / sample_size

# Validation accuracy
def validate(x_test, y_test, opt_x):
    predicted_labels = (np.sign(x_test @ opt_x) + 1) / 2
    errors = np.abs(predicted_labels - y_test)
    return 1 - np.mean(errors)

# SASS optimization on a single client
def sass_client(x, y, x_init, sample_size, local_epochs, alpha_0, gamma, theta, eps_f):
    x_current = x_init.copy()
    alpha = alpha_0
    lr_per_epoch = []
    grad_norm_per_epoch = []

    for _ in range(local_epochs):  # Perform multiple local epochs
        x, y = shuffle(x, y)
        grad = first_oracle(grad_f, x, y, x_current, sample_size)
        grad_norm = norm(grad)

        # Candidate update
        x_new = x_current - alpha * grad
        f_current = zeroth_oracle(f, x, y, x_current, sample_size)
        f_new = zeroth_oracle(f, x, y, x_new, sample_size)

        # Check modified Armijo condition
        if f_new <= f_current - alpha * theta * grad_norm**2 + 2 * eps_f:
            x_current = x_new  # Successful step
            alpha = alpha / gamma  # Increase step size
        else:
            alpha = alpha * gamma  # Decrease step size

        # Track metrics
        lr_per_epoch.append(alpha)
        grad_norm_per_epoch.append(grad_norm)

    return x_current, lr_per_epoch, grad_norm_per_epoch

# Federated Averaging: Aggregate models from clients
def federated_averaging(client_models):
    return np.mean(client_models, axis=0)

# Generate synthetic linearly separable global dataset
def generate_global_data(total_samples, n_features):
    X = np.random.randn(total_samples, n_features)
    true_weights = np.ones(n_features)  # Set weights to 1 for simplicity
    y = (np.sign(X @ true_weights) + 1) // 2  # Perfectly separable binary labels
    return X, y, true_weights

# Split global dataset among clients
def split_data_among_clients(X, y, n_clients):
    n_samples_per_client = len(X) // n_clients  # Divide data equally
    clients_data = []

    for i in range(n_clients):
        start_idx = i * n_samples_per_client
        end_idx = (i + 1) * n_samples_per_client
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        clients_data.append((X_client, y_client))
    
    return clients_data

# Main SASS + FedAvg implementation
n_clients = 3  # Number of clients
total_samples = 600  # Total number of samples
n_features = 20
sample_size = 32  # Batch size
global_epochs = 50  # Number of global epochs
local_epochs = 3  # Local epochs per client

# Generate a global dataset
X_global, y_global, true_weights = generate_global_data(total_samples, n_features)

# Split the global dataset among clients
clients_data = split_data_among_clients(X_global, y_global, n_clients)

# Test dataset (remaining portion of the global dataset for validation)
X_test, y_test = X_global[:100], y_global[:100]

# Initialize global model
global_model = np.zeros(n_features)
global_accuracy = []

# Metrics for clients
client_accuracies = [[] for _ in range(n_clients)]
client_lrs = [[] for _ in range(n_clients)]
client_grad_norms = [[] for _ in range(n_clients)]

# SASS parameters
alpha_0 = 1.0       # Original initial step size
gamma = 0.7         # Step size decrease factor
theta = 0.2         # Armijo condition constant
eps_f = 1e-2        # Noise constant

# Training loop
for epoch in range(global_epochs):
    local_models = []

    # Each client performs local SASS optimization
    for i in range(n_clients):
        X_client, y_client = clients_data[i]
        updated_model, lr_epoch, grad_norm_epoch = sass_client(
            X_client, y_client, global_model, sample_size, local_epochs,
            alpha_0, gamma, theta, eps_f
        )
        local_models.append(updated_model)
        client_lrs[i].append(np.mean(lr_epoch))  # Average learning rate per epoch
        client_grad_norms[i].append(np.mean(grad_norm_epoch))  # Average gradient norm
        client_acc = validate(X_test, y_test, updated_model)
        client_accuracies[i].append(client_acc)  # Accuracy of each client

    # Global aggregation
    global_model = federated_averaging(local_models)
    global_acc = validate(X_test, y_test, global_model)
    global_accuracy.append(global_acc)

    print(f"Epoch {epoch+1}/{global_epochs}, Global Accuracy: {global_acc:.4f}")

# Plot Accuracy of Each Client and Server
plt.figure()
for i in range(n_clients):
    plt.plot(range(global_epochs), client_accuracies[i], label=f"Client {i+1} Accuracy")
plt.plot(range(global_epochs), global_accuracy, label="Server (Global) Accuracy", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Client and Server per Epoch")
plt.legend()
plt.show()

# Plot Learning Rate per Client
plt.figure()
for i in range(n_clients):
    plt.plot(range(global_epochs), client_lrs[i], label=f"Client {i+1} Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Client per Epoch")
plt.legend()
plt.show()

# Plot Gradient Norm per Client
plt.figure()
for i in range(n_clients):
    plt.plot(range(global_epochs), client_grad_norms[i], label=f"Client {i+1} Gradient Norm")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm per Client per Epoch")
plt.legend()
plt.show()
