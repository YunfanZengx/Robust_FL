
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.utils import shuffle

EPS = 1e-6

# Logistic loss function
def f(xi, yi, opt_x):
    h_val = 1 / (1 + np.exp(-np.dot(xi, opt_x)))
    h_val = np.clip(h_val, EPS, 1 - EPS)  # Clip to prevent log(0)
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
    for _ in range(local_epochs):  # Perform multiple local epochs
        x, y = shuffle(x, y)
        grad = first_oracle(grad_f, x, y, x_current, sample_size)
        grad_norm = norm(grad)
        x_new = x_current - alpha * grad
        f_current = zeroth_oracle(f, x, y, x_current, sample_size)
        f_new = zeroth_oracle(f, x, y, x_new, sample_size)

        if f_new <= f_current - alpha * theta * grad_norm**2 + 2 * eps_f:
            x_current = x_new  # Successful step
            alpha = alpha / gamma  # Increase step size
        else:
            alpha = alpha * gamma  # Decrease step size

    return x_current

# Federated Geometric Median using Weiszfeld's algorithm
def weiszfeld_algorithm(client_models, tol=1e-4, max_iter=100):
    median = np.mean(client_models, axis=0)
                                                 
    for iteration in range(max_iter):
        # compute each client model's distance to the current estimate of the geometric median.
        distances = np.array([np.linalg.norm(model - median) for model in client_models]) 
        distances = np.clip(distances, a_min=1e-10, a_max=None)  # Avoid division by zero
        # closer model, higher weights
        weights = 1.0 / distances
        # weighted sum of the client models
        new_median = np.sum([w * model for w, model in zip(weights, client_models)], axis=0) / np.sum(weights)
        
        if np.linalg.norm(new_median - median) < tol:
            break
        
        median = new_median
    
    return median

# Generate synthetic linearly separable data
def generate_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    true_weights = np.ones(n_features)
    y = (np.sign(X @ true_weights) + 1) // 2
    return X, y, true_weights

# Data splitting functions for non-IID settings
def split_data_quantity_imbalance(X, y, n_clients, imbalance_factor):
    n_samples = len(X)
    min_samples = int((1 - imbalance_factor) * (n_samples / n_clients))
    max_samples = int((1 + imbalance_factor) * (n_samples / n_clients))

    clients_data = []
    for i in range(n_clients):
        n_client_samples = np.random.randint(min_samples, max_samples + 1)
        X_client, y_client = shuffle(X, y, n_samples=n_client_samples, random_state=i)
        clients_data.append((X_client, y_client))

    return clients_data

def split_data_distribution_imbalance(X, y, n_clients, label_distribution):
    unique_labels = np.unique(y)
    n_labels = len(unique_labels)

    if len(label_distribution) != n_clients or not all(len(dist) == n_labels for dist in label_distribution):
        raise ValueError("Label distribution must be a list of lists with length equal to the number of clients.")

    clients_data = [[] for _ in range(n_clients)]

    for label_idx, label in enumerate(unique_labels):
        label_indices = np.where(y == label)[0]
        np.random.shuffle(label_indices)
        split_indices = np.cumsum([int(len(label_indices) * label_distribution[i][label_idx]) for i in range(n_clients)])
        split_indices = [0] + list(split_indices[:-1]) + [len(label_indices)]

        for i in range(n_clients):
            start_idx, end_idx = split_indices[i], split_indices[i + 1]
            clients_data[i].append((X[label_indices[start_idx:end_idx]], y[label_indices[start_idx:end_idx]]))

    clients_data = [(np.concatenate([X for X, _ in data]), np.concatenate([y for _, y in data])) for data in clients_data]
    return clients_data

# Parameters
n_clients = 3
n_samples = 600
n_features = 20
sample_size = 32
global_epochs = 20
local_epochs = 3

# Generate dataset
X, y, _ = generate_data(n_samples, n_features)
X_test, y_test, _ = generate_data(500, n_features)

# Non-IID settings: Choose between quantity-based and distribution-based imbalance
imbalance_type = "distribution"  # Change to "quantity" for quantity-based imbalance

if imbalance_type == "quantity":
    imbalance_factor = 0.5
    clients_data = split_data_quantity_imbalance(X, y, n_clients, imbalance_factor)
else:
    label_distribution = [
        [0.8, 0.2],
        [0.5, 0.5],
        [0.2, 0.8]
    ]
    clients_data = split_data_distribution_imbalance(X, y, n_clients, label_distribution)

# Initialize global model
global_model = np.zeros(n_features)
global_accuracy = []
client_accuracies = [[] for _ in range(n_clients)]

# SASS parameters
alpha_0 = 1.0
gamma = 0.7
theta = 0.2
eps_f = 1e-2

# Training loop
for epoch in range(global_epochs):
    local_models = []

    for i in range(n_clients):
        X_client, y_client = clients_data[i]
        updated_model = sass_client(
            X_client, y_client, global_model, sample_size, local_epochs,
            alpha_0, gamma, theta, eps_f
        )
        local_models.append(updated_model)
        client_acc = validate(X_test, y_test, updated_model)
        client_accuracies[i].append(client_acc)

    # Use Weiszfeld's algorithm for geometric median
    global_model = weiszfeld_algorithm(local_models)
    global_acc = validate(X_test, y_test, global_model)
    global_accuracy.append(global_acc)

    print(f"Epoch {epoch+1}/{global_epochs}, Global Accuracy: {global_acc:.4f}")

# Plot accuracy
plt.figure()
for i in range(n_clients):
    plt.plot(range(global_epochs), client_accuracies[i], label=f"Client {i+1} Accuracy")
plt.plot(range(global_epochs), global_accuracy, label="Global Accuracy", linestyle="--", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy per Client and Global Model per Epoch (Weiszfeld's Algorithm)")
plt.legend()
plt.show()
