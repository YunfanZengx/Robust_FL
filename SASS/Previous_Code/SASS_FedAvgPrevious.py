import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt


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
# 3. Client Class
# ------------------------------
class Client:
    def __init__(self, client_id, data_loader, model, step_size_init, theta, gamma, eps_f):
        self.client_id = client_id
        self.data_loader = data_loader
        self.model = model
        self.step_size = step_size_init
        self.theta = theta
        self.gamma = gamma
        self.eps_f = eps_f
        self.step_sizes = []  # Track step sizes
        self.grad_norms = []  # Track gradient norms

    def local_update(self, epochs):
        self.model.train()
        local_model = self.model.state_dict()

        for epoch in range(epochs):
            for data, target in self.data_loader:
                grad_norm = 0.0
                loss_prev = 0.0

                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
                grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
                self.grad_norms.append(grad_norm.item())  # Log gradient norm

                # Propose step
                with torch.no_grad():
                    for param, grad in zip(self.model.parameters(), grads):
                        param -= self.step_size * grad

                # Check Armijo condition
                output_new = self.model(data)
                loss_new = nn.CrossEntropyLoss()(output_new, target)

                armijo_condition = loss_new <= loss - self.step_size * self.theta * (grad_norm ** 2) + 2 * self.eps_f
                if armijo_condition:
                    self.step_size = min(self.step_size / self.gamma, 1.0)
                else:
                    self.step_size = max(self.step_size * self.gamma, 1e-6)
                    self.model.load_state_dict(local_model)

                self.step_sizes.append(self.step_size)  # Log step size

        return self.model.state_dict(), self.step_size


# ------------------------------
# 4. Server Class
# ------------------------------
class Server:
    def __init__(self, global_model, clients):
        self.global_model = global_model
        self.clients = clients

    def aggregate(self, local_models):
        new_global_model = self.global_model.state_dict()
        for key in new_global_model.keys():
            new_global_model[key] = torch.mean(
                torch.stack([local_model[key] for local_model in local_models]), dim=0
            )
        self.global_model.load_state_dict(new_global_model)


# ------------------------------
# 5. Main Function
# ------------------------------
def main():
    # Parameters
    num_clients = 5
    rounds = 20
    local_epochs = 5
    step_size_init = 0.1
    theta, gamma, eps_f = 0.1, 1.1, 1e-4
    batch_size = 16

    # Global dataset
    global_dataset = GaussianBinaryDataset(n_samples=1000, dim=2)
    client_datasets = torch.utils.data.random_split(global_dataset, [200] * num_clients)

    # Initialize clients
    global_model = BinaryClassifier(input_dim=2)
    clients = []
    for i, dataset in enumerate(client_datasets):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_model = BinaryClassifier(input_dim=2)
        client = Client(i, data_loader, client_model, step_size_init, theta, gamma, eps_f)
        clients.append(client)

    server = Server(global_model, clients)
    accuracies = []

    # Federated training
    for round_idx in range(rounds):
        print(f"--- Round {round_idx+1} ---")
        local_models = []
        for client in clients:
            local_model, _ = client.local_update(local_epochs)
            local_models.append(local_model)
            print(f"Client {client.client_id} Final Step Size: {client.step_sizes[-1]:.6f}")

        # Aggregate
        server.aggregate(local_models)

        # Evaluate global model
        global_model.eval()
        test_loader = DataLoader(global_dataset, batch_size=100, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = global_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accuracy = correct / total * 100
        accuracies.append(accuracy)
        print(f"Global Model Accuracy: {accuracy:.2f}%\n")

    # Plot step sizes, gradient norms, and accuracy
    plt.figure(figsize=(10, 6))
    for client in clients:
        plt.plot(client.step_sizes, label=f"Client {client.client_id}")
    plt.title("Step Sizes Over Time")
    plt.xlabel("Iterations")
    plt.ylabel("Step Size")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    for client in clients:
        plt.plot(client.grad_norms, label=f"Client {client.client_id}")
    plt.title("Gradient Norms Over Time")
    plt.xlabel("Iterations")
    plt.ylabel("Gradient Norm")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, rounds + 1), accuracies, marker="o")
    plt.title("Global Model Accuracy Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy (%)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
