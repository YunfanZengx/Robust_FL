import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

#######################################################################
# 1) DATA PARTITIONING FUNCTIONS (IID and Dirichlet Non-IID)
#######################################################################
def iid_partition(full_dataset, num_clients, seed=42):
    """
    Splits the dataset into IID partitions for each client.

    Args:
        full_dataset (torch.utils.data.Dataset): The complete dataset.
        num_clients (int): Number of clients.
        seed (int): Random seed for reproducibility.

    Returns:
        List[torch.utils.data.Subset]: List of dataset subsets for each client.
    """
    np.random.seed(seed)
    dataset_size = len(full_dataset)
    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)

    samples_per_client = dataset_size // num_clients
    client_datasets = []
    start = 0
    for c in range(num_clients):
        if c < num_clients - 1:
            end = start + samples_per_client
        else:
            end = dataset_size  # Last client gets the remaining samples
        subset_indices = all_indices[start:end]
        subset_ds = Subset(full_dataset, subset_indices)
        client_datasets.append(subset_ds)
        start = end

    return client_datasets

def dirichlet_noniid_partition(full_dataset, num_clients, alpha=0.5, seed=42):
    """
    Splits the dataset into Non-IID partitions for each client based on a Dirichlet distribution.

    Args:
        full_dataset (torch.utils.data.Dataset): The complete dataset.
        num_clients (int): Number of clients.
        alpha (float): Parameter controlling the concentration of the Dirichlet distribution.
                       Lower alpha leads to more skewed distributions.
        seed (int): Random seed for reproducibility.

    Returns:
        List[torch.utils.data.Subset]: List of dataset subsets for each client.
    """
    np.random.seed(seed)
    labels = [sample[1] for sample in full_dataset]  # Assuming dataset returns (data, label)
    num_classes = len(set(labels))
    class_indices = {cls: np.where(np.array(labels) == cls)[0] for cls in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]

    for cls, indices in class_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet([alpha] * num_clients)
        # Scale proportions to the number of samples in this class
        proportions = np.array([p * len(indices) for p in proportions])
        proportions = proportions.astype(int)
        # Fix any rounding issues
        while proportions.sum() < len(indices):
            proportions[np.argmax(proportions)] += 1
        while proportions.sum() > len(indices):
            proportions[np.argmax(proportions)] -= 1
        start = 0
        for i, p in enumerate(proportions):
            client_indices[i].extend(indices[start:start + p].tolist())
            start += p

    # Create Subset objects
    client_datasets = [Subset(full_dataset, indices) for indices in client_indices]

    return client_datasets

#######################################################################
# 2) MODEL DEFINITION
#######################################################################
# class SmallNet(nn.Module):
#     """
#     A simple convolutional neural network for MNIST classification.
#     """
#     def __init__(self):
#         super(SmallNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Input channels=1 for MNIST
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)  # 10 classes for MNIST

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))  # Conv1 -> ReLU -> MaxPool
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))  # Conv2 -> ReLU -> MaxPool
#         x = x.view(-1, 320)  # Flatten
#         x = F.relu(self.fc1(x))  # FC1 -> ReLU
#         x = self.fc2(x)          # FC2
#         return F.log_softmax(x, dim=1)  # Log-Softmax for classification

class SmallNet(nn.Module):
    """
    This network has:
    - conv1: 5×5 kernel, in_channels=1, out_channels=6
    - conv2: 5×5 kernel, in_channels=6, out_channels=16
    - 2×2 max pool after each conv
    - fc1: 120 units
    - fc2: 84 units
    - final output: 10 (e.g., for MNIST)
    """
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)   # input: 1 channel, out: 6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # input: 6 channels, out: 16
        
        # After conv1 + pool, the image goes from 28×28 -> 24×24 -> 12×12
        # After conv2 + pool, the feature map goes from 12×12 -> 8×8 -> 4×4
        # So the flattened dimension is 16 * 4 * 4 = 256
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # conv1 -> ReLU -> 2×2 max pool
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # conv2 -> ReLU -> 2×2 max pool
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        
        # Flatten from (batch_size, 16, 4, 4) to (batch_size, 256)
        x = x.view(-1, 16*4*4)
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Final layer produces logits for 10 classes
        x = self.fc3(x)
        
        # Return log_softmax output for classification
        return F.log_softmax(x, dim=1)
#######################################################################
# 3) HELPER FUNCTIONS
#######################################################################
def train_one_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to train on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader, device):
    """
    Evaluates the model on the given dataloader.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the evaluation data.
        device (torch.device): Device to evaluate on.

    Returns:
        Tuple[float, float]: Average loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(dataloader.dataset)
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return test_loss, accuracy

def federated_average(client_models):
    """
    Performs Federated Averaging on the client models.

    Args:
        client_models (List[nn.Module]): List of client models.

    Returns:
        dict: Averaged state_dict.
    """
    avg_state_dict = copy.deepcopy(client_models[0].state_dict())
    for key in avg_state_dict.keys():
        for model in client_models[1:]:
            avg_state_dict[key] += model.state_dict()[key]
        avg_state_dict[key] = torch.div(avg_state_dict[key], len(client_models))
    return avg_state_dict

#######################################################################
# 4) MAIN FEDERATED SCRIPT
#######################################################################
def main():
    # Federated Learning Parameters
    num_clients = 2
    num_global_rounds = 100
    local_epochs = 1
    batch_size = 32
    learning_rate = 0.01
    momentum = 0.9
    partition_strategy = 'IID'  # Choose between 'IID' and 'Non-IID'

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Save path for plots
    save_path = "./fedAvg_SGD_results"
    os.makedirs(save_path, exist_ok=True)

    # Data setup
    data_dir = './mnist_data'
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_dataset = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

    # Data partitioning
    if partition_strategy == 'IID':
        client_datasets = iid_partition(full_train_dataset, num_clients, seed=42)
    elif partition_strategy == 'Non-IID':
        client_datasets = dirichlet_noniid_partition(full_train_dataset, num_clients, alpha=0.5, seed=42)
    else:
        raise ValueError("Invalid partition strategy. Choose 'IID' or 'Non-IID'.")

    # Initialize global model
    global_model = SmallNet().to(device)
    global_model_weights = copy.deepcopy(global_model.state_dict())

    # Tracking metrics
    train_losses_over_rounds = []
    test_losses_over_rounds = []
    global_accuracies = []
    client_accuracies = [[] for _ in range(num_clients)]

    for round_idx in range(num_global_rounds):
        print(f"\n--- Global Round {round_idx+1}/{num_global_rounds} ---")
        client_models = []
        local_train_losses = []

        for client_idx in range(num_clients):
            # Initialize client model with global weights
            client_model = SmallNet().to(device)
            client_model.load_state_dict(copy.deepcopy(global_model_weights))

            # Define optimizer
            optimizer = torch.optim.SGD(client_model.parameters(), lr=learning_rate, momentum=momentum)

            # Define dataloader for client
            local_loader = DataLoader(client_datasets[client_idx], batch_size=batch_size, shuffle=True)

            # Train local model
            local_train_loss = 0.0
            for epoch in range(local_epochs):
                epoch_loss = train_one_epoch(client_model, local_loader, optimizer, device)
                local_train_loss += epoch_loss
            avg_local_train_loss = local_train_loss / local_epochs
            local_train_losses.append(avg_local_train_loss)

            # Evaluate on client data
            client_eval_loader = DataLoader(client_datasets[client_idx], batch_size=500, shuffle=False)
            _, client_accuracy = evaluate(client_model, client_eval_loader, device)
            client_accuracies[client_idx].append(client_accuracy)

            # Append to list of client models
            client_models.append(client_model)

            print(f"Client {client_idx+1} - Training Loss: {avg_local_train_loss:.4f}, Accuracy: {client_accuracy:.2f}%")

        # Federated Averaging
        global_model_weights = federated_average(client_models)
        global_model.load_state_dict(global_model_weights)

        # Evaluate global model on test data
        test_loss, test_accuracy = evaluate(global_model, test_loader, device)
        test_losses_over_rounds.append(test_loss)
        global_accuracies.append(test_accuracy)

        print(f"Global Model - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        train_losses_over_rounds.append(np.mean(local_train_losses))

    # Plotting
    plt.figure(figsize=(10, 7))
    for client_idx in range(num_clients):
        plt.plot(
            range(1, num_global_rounds + 1),
            client_accuracies[client_idx],
            marker='o',
            linestyle='-',
            label=f"Client {client_idx+1} Accuracy"
        )
    plt.plot(
        range(1, num_global_rounds + 1),
        global_accuracies,
        marker='s',
        linestyle='--',
        color='red',
        label="Global Model Accuracy"
    )
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Client and Global Model Accuracy per Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "client_and_global_accuracy.png"))
    plt.close()

    print("\nTraining completed. Plots saved in:", save_path)

if __name__ == "__main__":
    main()
