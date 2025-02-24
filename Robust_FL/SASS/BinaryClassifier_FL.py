import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------
# Data Generation and Splitting
# ---------------------------
def generate_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    class_0 = np.random.normal(1, 1, (num_samples // 2, 2))  # Mean=1, Std=1
    class_1 = np.random.normal(4, 1, (num_samples // 2, 2))  # Mean=4, Std=1
    labels_0 = np.zeros(num_samples // 2)
    labels_1 = np.ones(num_samples // 2)
    X = np.vstack((class_0, class_1))
    y = np.hstack((labels_0, labels_1))
    return X, y

def split_data_to_clients(X, y, client_class0, client_class1):
    class0_idx = np.where(y == 0)[0]
    class1_idx = np.where(y == 1)[0]
    np.random.shuffle(class0_idx)
    np.random.shuffle(class1_idx)

    clients_data = []
    c0_ptr = 0
    c1_ptr = 0
    for c0_count, c1_count in zip(client_class0, client_class1):
        c0_indices = class0_idx[c0_ptr : c0_ptr + c0_count]
        c1_indices = class1_idx[c1_ptr : c1_ptr + c1_count]
        c0_ptr += c0_count
        c1_ptr += c1_count
        
        client_indices = np.concatenate([c0_indices, c1_indices])
        np.random.shuffle(client_indices)
        X_client = X[client_indices]
        y_client = y[client_indices]
        clients_data.append((X_client, y_client))

    return clients_data

# ---------------------------
# Logistic Regression Model
# ---------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# ---------------------------
# Client Class
# ---------------------------
class Client:
    def __init__(self, X, y, input_dim, learning_rate=0.1, local_epochs=5, batch_size=32):
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                     torch.tensor(y, dtype=torch.float32).view(-1,1))
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = LogisticRegressionModel(input_dim)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

    def set_parameters(self, global_params):
        self.model.load_state_dict(global_params)

    def train(self):
        self.model.train()
        for epoch in range(self.local_epochs):
            for X_batch, y_batch in self.dataloader:
                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

    def get_parameters(self):
        return self.model.state_dict()
    
    def evaluate(self, dataset=None):
        if dataset is None:
            dataset = self.dataset
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = self.model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = (y_pred >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

# ---------------------------
# Server Class
# ---------------------------
class Server:
    def __init__(self, input_dim, clients):
        self.global_model = LogisticRegressionModel(input_dim)
        self.clients = clients
        self.criterion = nn.BCELoss()

    def aggregate(self, client_params_list):
        global_params = self.global_model.state_dict()
        for key in global_params.keys():
            global_params[key] = torch.mean(torch.stack([params[key] for params in client_params_list]), dim=0)
        self.global_model.load_state_dict(global_params)

    def federated_train(self, X_test, y_test, rounds=3):
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.float32).view(-1,1))
        client_accuracies_last_round = []
        server_accuracy_last_round = None

        for r in range(rounds):
            print(f"--- Federated Round {r + 1} ---")
            global_params = self.global_model.state_dict()
            client_params_list = []

            client_test_accuracies = []

            for i, client in enumerate(self.clients):
                client.set_parameters(global_params)
                client.train()
                client_params_list.append(client.get_parameters())

                # Evaluate client metrics
                _, test_acc = client.evaluate()
                client_test_accuracies.append(test_acc)

            self.aggregate(client_params_list)

            # Evaluate server accuracy on test dataset
            if len(X_test) > 0:
                _, server_test_acc = self.evaluate(test_dataset)
            else:
                server_test_acc = 0

            if r == rounds - 1:  # Save accuracies from the last round
                client_accuracies_last_round = client_test_accuracies
                server_accuracy_last_round = server_test_acc

            print("Client Test Accuracies:", client_test_accuracies)
            print(f"Server Test Accuracy: {server_test_acc}")
            print()

        return client_accuracies_last_round, server_accuracy_last_round

    def evaluate(self, dataset):
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        self.global_model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                y_pred = self.global_model(X_batch)
                loss = self.criterion(y_pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = (y_pred >= 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
        avg_loss = total_loss / total
        accuracy = correct / total
        return avg_loss, accuracy

# ---------------------------
# Experiment Function
# ---------------------------
def run_experiment(num_clients, client_class0, client_class1, num_samples=1000, test_size=0.2, rounds=3, seed=42):
    # Generate data
    X, y = generate_data(num_samples=num_samples, seed=seed)
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Count class 0 and class 1 in the training set
    class0_count = np.sum(y_train == 0)
    class1_count = np.sum(y_train == 1)

    # Check and adjust client distributions
    if sum(client_class0) != class0_count or sum(client_class1) != class1_count:
        adjusted_client_class0 = np.array(client_class0) / sum(client_class0) * class0_count
        adjusted_client_class1 = np.array(client_class1) / sum(client_class1) * class1_count
        adjusted_client_class0 = np.round(adjusted_client_class0).astype(int)
        adjusted_client_class1 = np.round(adjusted_client_class1).astype(int)
        adjusted_client_class0[-1] += class0_count - np.sum(adjusted_client_class0)
        adjusted_client_class1[-1] += class1_count - np.sum(adjusted_client_class1)
        client_class0 = adjusted_client_class0.tolist()
        client_class1 = adjusted_client_class1.tolist()

    # Split data among clients
    clients_data = split_data_to_clients(X_train, y_train, client_class0, client_class1)
    clients = []
    input_dim = X_train.shape[1]
    for (Xc, yc) in clients_data:
        client = Client(Xc, yc, input_dim, learning_rate=0.1, local_epochs=5, batch_size=32)
        clients.append(client)

    server = Server(input_dim, clients)
    print(f"Running experiment with {num_clients} clients")
    client_accuracies, server_accuracy = server.federated_train(X_test, y_test, rounds=rounds)

    # Create the figure
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # Bar plot for sample distribution
    bar_width = 0.4
    indices = np.arange(num_clients)
    bars1 = ax1.bar(indices, client_class0, bar_width, label="Class 0 Samples", color='blue', alpha=0.7)
    bars2 = ax1.bar(indices, client_class1, bar_width, bottom=client_class0, label="Class 1 Samples", color='orange', alpha=0.7)
    ax1.set_ylabel("Number of Samples", fontsize=14)
    ax1.set_xlabel("Clients", fontsize=14)
    ax1.set_xticks(indices)
    ax1.set_xticklabels([f"Client {i+1}" for i in range(num_clients)], fontsize=12)
    ax1.set_title(
        f"Sample Distribution and Test Accuracy\n"
        f"Clients: {num_clients}, Training Samples: {len(X_train)}, Class 0: {class0_count}, Class 1: {class1_count}",
        fontsize=16
    )

    # Line plot for accuracies
    ax2 = ax1.twinx()
    line_clients, = ax2.plot(indices, client_accuracies, 'o-', color='green', label="Client Accuracy")
    line_server = ax2.axhline(y=server_accuracy, color='red', linestyle='--', label="Server Accuracy")
    ax2.set_ylabel("Accuracy", fontsize=14)
    ax2.set_ylim(0, 1)

    # Annotate accuracy values
    for i, acc in enumerate(client_accuracies):
        ax2.text(i, acc + 0.02, f"{acc:.2f}", color='green', ha='center', fontsize=12)
    ax2.text(indices[-1] + 0.5, server_accuracy + 0.02, f"{server_accuracy:.2f}", color='red', ha='center', fontsize=12)

    # Add a combined legend
    handles = [bars1, bars2, line_clients, line_server]
    labels = ["Class 0 Samples", "Class 1 Samples", "Client Accuracy", "Server Accuracy"]
    fig.legend(
        handles, labels,
        loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12, ncol=2, frameon=False
    )

    # Adjust layout to leave space for the legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()



# ---------------------------
# Main Program
# ---------------------------
if __name__ == "__main__":
    total_samples = 1000  # Fixed total number of samples

    for num_clients in [1, 4, 6]:  # Experiment with 1, 4, and 6 clients
        print(f"\nRunning experiments with {num_clients} clients and total {total_samples} samples")

        # Case 1: Each client has the same number of data from Class 0 and Class 1
        print("\nCase 1: Each client has the same number of data from Class 0 and Class 1")
        base_size = total_samples // num_clients
        client_sizes = [base_size] * num_clients
        client_sizes[-1] += total_samples - sum(client_sizes)  # Adjust last client to ensure total matches
        client_class0 = [size // 2 for size in client_sizes]
        client_class1 = [size // 2 for size in client_sizes]
        run_experiment(
            num_clients=num_clients,
            client_class0=client_class0,
            client_class1=client_class1,
            num_samples=total_samples,
            rounds=3
        )

        # Case 2: Each client has the same proportion of data from Class 0 and Class 1, but different amounts
        print("\nCase 2: Each client has the same proportion of data from Class 0 and Class 1, but different amounts")
        proportion_sizes = [10, 20, 50, 100, 150, 200][:num_clients]
        total_proportion = sum(proportion_sizes)
        client_sizes = [int(size / total_proportion * total_samples) for size in proportion_sizes]
        client_sizes[-1] += total_samples - sum(client_sizes)  # Adjust last client
        client_class0 = [size // 2 for size in client_sizes]
        client_class1 = [size // 2 for size in client_sizes]
        run_experiment(
            num_clients=num_clients,
            client_class0=client_class0,
            client_class1=client_class1,
            num_samples=total_samples,
            rounds=3
        )

        # Case 3: Each client has extremely imbalanced proportions, but the same total number of samples
        print("\nCase 3: Each client has extremely imbalanced proportions, but the same total number of samples")
        base_size = total_samples // num_clients
        client_sizes = [base_size] * num_clients
        client_sizes[-1] += total_samples - sum(client_sizes)  # Adjust last client to ensure total matches
        imbalance_ratios = [(90, 10), (10, 90), (80, 20), (20, 80), (70, 30), (30, 70)][:num_clients]
        client_class0 = [int(size * ratio[0] / 100) for size, ratio in zip(client_sizes, imbalance_ratios)]
        client_class1 = [size - c0 for size, c0 in zip(client_sizes, client_class0)]
        run_experiment(
            num_clients=num_clients,
            client_class0=client_class0,
            client_class1=client_class1,
            num_samples=total_samples,
            rounds=3
        )

        # Case 4: Each client has imbalanced proportions and different amounts of samples
        print("\nCase 4: Each client has imbalanced proportions and different amounts of samples")
        proportion_sizes = [10, 20, 50, 100, 150, 200][:num_clients]
        total_proportion = sum(proportion_sizes)
        client_sizes = [int(size / total_proportion * total_samples) for size in proportion_sizes]
        client_sizes[-1] += total_samples - sum(client_sizes)  # Adjust last client
        imbalance_ratios = [(90, 10), (10, 90), (80, 20), (20, 80), (70, 30), (30, 70)][:num_clients]
        client_class0 = [int(size * ratio[0] / 100) for size, ratio in zip(client_sizes, imbalance_ratios)]
        client_class1 = [size - c0 for size, c0 in zip(client_sizes, client_class0)]
        run_experiment(
            num_clients=num_clients,
            client_class0=client_class0,
            client_class1=client_class1,
            num_samples=total_samples,
            rounds=3
        )
