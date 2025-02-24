import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from sklearn.utils import shuffle

EPS = 1e-6

# Zeroth order oracle
def f(xi, yi, opt_x):
    h_val = 1.0 / (1.0 + np.exp(-np.dot(xi, opt_x)))
    h_val = np.clip(h_val, EPS, 1 - EPS)
    return -yi * np.log(h_val) - (1 - yi) * np.log(1 - h_val)

# First-order gradient
def grad_f(xi, yi, opt_x):
    h_val = 1.0 / (1.0 + np.exp(-np.dot(xi, opt_x)))
    return -(yi - h_val) * xi

# Compute true function value
def true_function_val(f, x, y, opt_x):
    f_val = 0.0
    for ind in range(len(x)):
        f_val += f(x[ind], y[ind], opt_x)
    return f_val / len(x)

# Validation accuracy
def validate(f, x_test, y_test, x_current):
    predicted_labels = (np.sign(x_test @ x_current) + 1) / 2
    errors = np.abs(predicted_labels - y_test)
    return 1 - np.mean(errors)

# Zeroth-order oracle with sampling
def zeroth_oracle(f, x, y, opt_x, sample_size):
    sample_list = np.random.choice(range(len(x)), sample_size, replace=False)
    f_noisy_val = 0.0
    for ind in sample_list:
        f_noisy_val += f(x[ind], y[ind], opt_x)
    return f_noisy_val / sample_size

# First-order oracle with sampling
def first_oracle(f, grad_f, x, y, opt_x, subsample_num):
    grad_list = np.random.choice(range(len(x)), subsample_num, replace=False)
    grad_noisy = np.zeros(len(opt_x))
    for ind in grad_list:
        grad_noisy += grad_f(x[ind], y[ind], opt_x)
    return grad_noisy / subsample_num

# Estimate noise in function value
def estimate_epi_f(f, x, y, opt_x, zeroth_oracle, sample_size, n_trials, factor=1/5):
    result_arr = np.zeros(n_trials)
    for i in range(n_trials):
        result_arr[i] = zeroth_oracle(f, x, y, opt_x, sample_size)
    return np.std(result_arr) * factor

# Line search with same batch
def line_search_same_batch(f, grad_f, x, y, zeroth_oracle, first_oracle, sample_size, x_0,
                           eps_f, alpha_0, alpha_max, dec_gamma, inc_gamma, theta, epoch,
                           num_epochs_per_estimate=1, factor=1/5, epi_f_zeroth_oracle=zeroth_oracle, x_test=None, y_test=None):
    dim = len(x_0)
    total_number = len(x)
    x_current = x_0
    iteration = epoch * (total_number // sample_size)
    
    fun_val_arr = np.zeros(iteration)
    val_acc_arr = np.zeros(iteration)
    alpha_arr = np.zeros(iteration)
    grad_norm_arr = np.zeros(iteration)
    
    epoch_acc = []
    epoch_lr = []
    
    alpha = alpha_0
    iteration_count = 0
    start_ind = 0
    prev_epoch_number = -1
    epoch_learning_rates = []
    epoch_accuracies = []
    
    while iteration_count < iteration:
        if start_ind == 0:
            x, y = shuffle(x, y, random_state=iteration_count)
        
        epoch_number = int(np.floor(iteration_count * sample_size / total_number))
        if epoch_number == int(prev_epoch_number + 1) and (epoch_number % num_epochs_per_estimate == 0):
            prev_epoch_number += 1
            eps_f = estimate_epi_f(f, x, y, x_current, epi_f_zeroth_oracle, sample_size, n_trials=30, factor=factor)
        
        grad_approximation = first_oracle(f, grad_f, x, y, x_current, sample_size)
        grad_norm = LA.norm(grad_approximation)
        grad_norm_arr[iteration_count] = grad_norm
        alpha_arr[iteration_count] = alpha
        
        x_new = x_current - alpha * grad_approximation
        f_current = zeroth_oracle(f, x, y, x_current, sample_size)
        f_new = zeroth_oracle(f, x, y, x_new, sample_size)
        
        if f_new <= f_current - alpha * theta * grad_norm ** 2 + 2 * eps_f:
            x_current = x_new
            alpha = min(inc_gamma * alpha, alpha_max)
        else:
            alpha = dec_gamma * alpha
        
        fun_val_arr[iteration_count] = true_function_val(f, x, y, x_current)
        val_acc_arr[iteration_count] = validate(f, x_test, y_test, x_current)
        
        # Track epoch-wise metrics
        epoch_learning_rates.append(alpha)
        epoch_accuracies.append(val_acc_arr[iteration_count])
        
        # Aggregate at the end of each epoch
        if (iteration_count + 1) % (total_number // sample_size) == 0:
            epoch_acc.append(np.mean(epoch_accuracies))
            epoch_lr.append(np.mean(epoch_learning_rates))
            epoch_learning_rates = []
            epoch_accuracies = []
        
        start_ind = (start_ind + sample_size) % total_number
        iteration_count += 1
    
    return x_current, fun_val_arr, val_acc_arr, alpha_arr, grad_norm_arr, epoch_acc, epoch_lr

# Generate synthetic data
def generate_data(n_samples, n_features):
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features)
    y = (np.sign(X @ true_weights) + 1) // 2
    return X, y, true_weights

# Main experiment
n_samples = 1000
n_features = 20
X, y, true_weights = generate_data(n_samples, n_features)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

x_0 = np.zeros(n_features)
sample_size = 128
epoch = 100
eps_f = 1e-2
alpha_0 = 1
alpha_max = 10
dec_gamma = 0.7
inc_gamma = 1.25
theta = 0.2

x_opt, fun_val_arr, val_acc_arr, alpha_arr, grad_norm_arr, epoch_acc, epoch_lr = line_search_same_batch(
    f, grad_f, X_train, y_train, zeroth_oracle, first_oracle, sample_size, x_0,
    eps_f, alpha_0, alpha_max, dec_gamma, inc_gamma, theta, epoch, x_test=X_test, y_test=y_test
)

# Plot results

# Plot learning rate per iteration
plt.figure()
plt.plot(range(len(alpha_arr)), alpha_arr, label="Learning Rate per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Iteration")
plt.legend()
plt.show()

# Plot gradient norm per iteration
plt.figure()
plt.plot(range(len(grad_norm_arr)), grad_norm_arr, label="Gradient Norm per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm per Iteration")
plt.legend()
plt.show()

# Plot accuracy per epoch
plt.figure()
plt.plot(range(len(epoch_acc)), epoch_acc, label="Validation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Accuracy per Epoch")
plt.legend()
plt.show()

# Plot learning rate per epoch
plt.figure()
plt.plot(range(len(epoch_lr)), epoch_lr, label="Learning Rate per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Epoch")
plt.legend()
plt.show()