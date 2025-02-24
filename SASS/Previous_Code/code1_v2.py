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

from sass import Sass  # Your SASS optimizer

########################################################################
# 1) DATA PARTITIONING FUNCTIONS (IID and Dirichlet Non-IID, from code2)
########################################################################
def iid_partition(full_dataset, num_clients, seed=42):
    """
    Returns a list of Subset datasets, each an IID partition of full_dataset.
    Each client gets exactly len(full_dataset)//num_clients samples
    (with any remainder going to the last client).
    """
    np.random.seed(seed)
    
    dataset_size = len(full_dataset)
    all_indices = np.arange(dataset_size)
    np.random.shuffle(all_indices)

    # If you want exact equal splits, do this:
    samples_per_client = dataset_size // num_clients
    client_datasets = []
    start = 0
    for c in range(num_clients):
        if c < num_clients - 1:
            end = start + samples_per_client
        else:
            end = dataset_size  # last client gets leftover
        subset_indices = all_indices[start:end]
        subset_ds = Subset(full_dataset, subset_indices)
        client_datasets.append(subset_ds)
        start = end

    return client_datasets

def dirichlet_noniid_partition(full_dataset, num_clients, alpha=0.5, seed=42):
    """
    Dirichlet-based Non-IID partition.
      - alpha controls class distribution skew (lower=more skewed).
      - We assume MNIST-like classification (0..9 classes).
    """
    np.random.seed(seed)

    # Sort indices by class
    labels = [sample[1] for sample in full_dataset]  # (data, label) pairs
    num_classes = len(set(labels))
    class_indices = {cls: [] for cls in range(num_classes)}
    
    for idx, lab in enumerate(labels):
        class_indices[lab].append(idx)
    
    # Shuffle each class’ indices
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # Dirichlet draws: each class’s samples are distributed among clients
    # according to a Dirichlet distribution over 'num_clients'.
    client_class_proportions = np.random.dirichlet([alpha]*num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]
    
    for cls_idx in range(num_classes):
        cindices = class_indices[cls_idx]
        if len(cindices) == 0:
            continue
        # # samples from this class
        num_cls_samples = len(cindices)
        
        # how many samples go to each client
        proportions = client_class_proportions[cls_idx]
        # “floor” or “round” them so total = num_cls_samples
        # a more direct way is to do a multinomial draw
        class_split = (proportions * num_cls_samples).astype(int)
        
        # Fix any rounding issues (leftover) so total == num_cls_samples
        while sum(class_split) < num_cls_samples:
            i = np.random.choice(num_clients)
            class_split[i] += 1
        while sum(class_split) > num_cls_samples:
            i = np.random.choice(num_clients)
            if class_split[i] > 0:
                class_split[i] -= 1
        
        # Distribute indices accordingly
        idx_start = 0
        for client_id, split_count in enumerate(class_split):
            idx_end = idx_start + split_count
            assigned_indices = cindices[idx_start:idx_end]
            client_indices[client_id].extend(assigned_indices)
            idx_start = idx_end

    # Create the Subset objects
    client_datasets = []
    for c in range(num_clients):
        ds = Subset(full_dataset, client_indices[c])
        client_datasets.append(ds)

    return client_datasets

########################################################################
# 2) MODEL DEFINITION (SAME AS code 1_v1)
########################################################################
class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

########################################################################
# 3) HELPER FUNCTIONS
########################################################################
def train_one_epoch(model, dataloader, optimizer):
    """
    Runs one epoch of training using the SASS optimizer.
    Returns the average loss over this epoch.
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def evaluate(model, dataloader):
    """
    Evaluate the model on a given dataloader (e.g., test set).
    Returns average loss and accuracy.
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return test_loss, accuracy

def average_weights(list_of_models):
    """
    Simple FedAvg aggregator: 
    Takes a list of model replicas and returns an *in-place* averaged state_dict.
    """
    base_model = list_of_models[0]
    base_params = base_model.state_dict()
    
    for key in base_params.keys():
        for i in range(1, len(list_of_models)):
            base_params[key] += list_of_models[i].state_dict()[key]
        base_params[key] = base_params[key] / len(list_of_models)
    
    return base_params

########################################################################
# 4) MAIN FEDERATED SCRIPT
########################################################################
def main():
    # Federated parameters
    num_clients = 2
    num_global_rounds = 100
    local_epochs = 1

    # Save path
    save_path = "./new_Fed_sass_v2_result_2_clients"
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

    #################################################################
    # CHOOSE ONE PARTITION STRATEGY: IID or DIRICHLET NON-IID
    #################################################################
    # Example: IID
    client_datasets = iid_partition(full_train_dataset, num_clients=num_clients, seed=42)

    # Example: Dirichlet Non-IID
    #client_datasets = dirichlet_noniid_partition(full_train_dataset, num_clients=num_clients, alpha=0.2, seed=42)

    # Initialize global model
    global_model = SmallNet()
    global_model_weights = copy.deepcopy(global_model.state_dict())

    # Tracking stats
    train_losses_over_rounds = []
    test_losses_over_rounds = []
    accuracy_over_rounds = []
    # We track each client's LR per round
    lr_per_client_per_round = [[] for _ in range(num_clients)]

    for round_idx in range(num_global_rounds):
        print(f"\n--- Global Round {round_idx+1}/{num_global_rounds} ---")
        
        local_models = []
        local_avg_losses = []

        for client_idx in range(num_clients):
            # 1) local model
            local_model = SmallNet()
            local_model.load_state_dict(copy.deepcopy(global_model_weights))

            # 2) dataloader
            local_loader = DataLoader(client_datasets[client_idx], batch_size=32, shuffle=True)

            # 3) SASS
            n_batches_per_epoch = len(local_loader)
            local_optimizer = Sass(local_model.parameters(), n_batches_per_epoch=n_batches_per_epoch)

            # 4) Local training
            for ep in range(local_epochs):
                avg_loss = train_one_epoch(local_model, local_loader, local_optimizer)

            local_models.append(local_model)
            local_avg_losses.append(avg_loss)

            # 5) Check final LR used by this client
            if local_optimizer.state['step_size_vs_nn_passes']:
                last_step_size = local_optimizer.state['step_size_vs_nn_passes'][-1][1]
            else:
                last_step_size = float('nan')
            lr_per_client_per_round[client_idx].append(last_step_size)

        # FedAvg aggregator
        if num_clients > 1:
            new_weights = average_weights(local_models)
            global_model.load_state_dict(new_weights)
            global_model_weights = copy.deepcopy(new_weights)
        else:
            global_model.load_state_dict(local_models[0].state_dict())
            global_model_weights = copy.deepcopy(local_models[0].state_dict())

        # Evaluate
        mean_train_loss = sum(local_avg_losses) / len(local_avg_losses)
        train_losses_over_rounds.append(mean_train_loss)

        test_loss, test_acc = evaluate(global_model, test_loader)
        test_losses_over_rounds.append(test_loss)
        accuracy_over_rounds.append(test_acc)

        print(f"Round {round_idx+1} - Avg local train loss: {mean_train_loss:.4f}")
        print(f"Round {round_idx+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

    # PLOTS
    # (1) Plot average local train loss across global rounds
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_global_rounds + 1), train_losses_over_rounds, 'o-b')
    plt.xlabel("Global Round")
    plt.ylabel("Average Local Training Loss")
    plt.title("Federated - Avg Local Training Loss vs. Global Round")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "federated_train_loss.png"))
    plt.close()

    # (2) Plot test accuracy vs global rounds
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_global_rounds + 1), accuracy_over_rounds, 'o-g')
    plt.xlabel("Global Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Federated - Global Model Test Accuracy vs. Global Round")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "federated_test_accuracy.png"))
    plt.close()

    # (3) Plot each client's LR vs. global rounds
    plt.figure(figsize=(8, 6))
    for client_idx in range(num_clients):
        plt.plot(
            range(1, num_global_rounds+1),
            lr_per_client_per_round[client_idx],
            marker='o',
            linestyle='-',
            label=f"Client {client_idx}"
        )
    plt.xlabel("Global Round")
    plt.ylabel("Last Step Size (SASS)")
    plt.title("Step Size per Client vs. Global Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "lr_per_client_per_round.png"))
    plt.close()

    print("\nTraining completed. Plots saved in:", save_path)

if __name__ == "__main__":
    main()
