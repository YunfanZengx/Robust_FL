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


from sass import Sass

def iid_partition(full_dataset, num_clients, seed=42):
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
    
    'seed' is set for reproducibility.
    """
    np.random.seed(seed)

    labels = [sample[1] for sample in full_dataset]  # each sample is (data, label)
    num_classes = len(set(labels))
    class_indices = {cls: [] for cls in range(num_classes)}
    
    # Group indices by class
    for idx, lab in enumerate(labels):
        class_indices[lab].append(idx)
    
    # Shuffle each class’s indices
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # Draw Dirichlet proportions
    client_class_proportions = np.random.dirichlet([alpha]*num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]
    
    for cls_idx in range(num_classes):
        cindices = class_indices[cls_idx]
        if len(cindices) == 0:
            continue
        num_cls_samples = len(cindices)
        proportions = client_class_proportions[cls_idx]
        class_split = (proportions * num_cls_samples).astype(int)
        
        # Fix rounding so total == num_cls_samples
        while sum(class_split) < num_cls_samples:
            i = np.random.choice(num_clients)
            class_split[i] += 1
        while sum(class_split) > num_cls_samples:
            i = np.random.choice(num_clients)
            if class_split[i] > 0:
                class_split[i] -= 1
        
        idx_start = 0
        for client_id, split_count in enumerate(class_split):
            idx_end = idx_start + split_count
            assigned_indices = cindices[idx_start:idx_end]
            client_indices[client_id].extend(assigned_indices)
            idx_start = idx_end

    client_datasets = []
    for c in range(num_clients):
        ds = Subset(full_dataset, client_indices[c])
        client_datasets.append(ds)

    return client_datasets

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

        # SASS step, which calls closure internally
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


def Federated_Average(list_of_models):

    base_model = list_of_models[0]
    base_params = base_model.state_dict()
    
    for key in base_params.keys():
        for i in range(1, len(list_of_models)):
            base_params[key] += list_of_models[i].state_dict()[key]
        base_params[key] = base_params[key] / len(list_of_models)
    
    return base_params



def main():

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Federated parameters
    num_clients = 2
    num_global_rounds = 100
    local_epochs = 1

    # Create directory to save results
    save_path = "./new_Fed_sass_v4_result_2_clients"
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

    # Partition data among clients
    client_datasets = iid_partition(full_train_dataset, num_clients=num_clients, seed=seed)

    # Or for Dirichlet Non-IID:
    # client_datasets = dirichlet_noniid_partition(
    #     full_train_dataset, num_clients=num_clients, alpha=0.2, seed=seed
    # )

    # Initialize global model
    global_model = SmallNet()
    global_model_weights = copy.deepcopy(global_model.state_dict())

    # Stats trackers
    train_losses_over_rounds = []
    test_losses_over_rounds = []
    accuracy_over_rounds = []

    # Accuracy per client across global rounds
    accuracy_per_client_over_rounds = [[] for _ in range(num_clients)]

    # LR per client per global round
    lr_per_client_per_round = [[] for _ in range(num_clients)]


    # Compute data distribution for each client
    num_classes = 10  # MNIST
    client_data_distribution = []
    for client_idx in range(num_clients):
        labels = [full_train_dataset.targets[idx].item() for idx in client_datasets[client_idx].indices]
        class_counts = [labels.count(c) for c in range(num_classes)]
        client_data_distribution.append(class_counts)

    # Federated rounds
    for round_idx in range(num_global_rounds):
        print(f"\n--- Global Round {round_idx+1}/{num_global_rounds} ---")
        
        local_models = []
        local_avg_losses = []

        for client_idx in range(num_clients):
            # 1) Create local model
            local_model = SmallNet()
            local_model.load_state_dict(copy.deepcopy(global_model_weights))

            # 2) Create local dataloader
            local_loader = DataLoader(client_datasets[client_idx], batch_size=32, shuffle=True)

            # 3) SASS optimizer
            n_batches_per_epoch = len(local_loader)
            local_optimizer = Sass(local_model.parameters(), n_batches_per_epoch=n_batches_per_epoch)
            
            # 4) Local training (for 'local_epochs')
            for ep in range(local_epochs):
                avg_loss = train_one_epoch(local_model, local_loader, local_optimizer)

            local_models.append(local_model)
            local_avg_losses.append(avg_loss)

            # 5) Evaluate on client data
            client_loader = DataLoader(client_datasets[client_idx], batch_size=500, shuffle=False)
            client_loss, client_acc = evaluate(local_model, client_loader)
            accuracy_per_client_over_rounds[client_idx].append(client_acc)

            # 6) Check final LR used by this client in the last step
            if local_optimizer.state['step_size_vs_nn_passes']:
                last_step_size = local_optimizer.state['step_size_vs_nn_passes'][-1][1]
            else:
                last_step_size = float('nan')
            lr_per_client_per_round[client_idx].append(last_step_size)

        # 7) Federated Average aggregator
        if num_clients > 1:
            new_weights = Federated_Average(local_models)
            global_model.load_state_dict(new_weights)
            global_model_weights = copy.deepcopy(new_weights)
        else:
            global_model.load_state_dict(local_models[0].state_dict())
            global_model_weights = copy.deepcopy(local_models[0].state_dict())

        # Evaluate global model on test
        mean_train_loss = sum(local_avg_losses) / len(local_avg_losses)
        train_losses_over_rounds.append(mean_train_loss)

        test_loss, test_acc = evaluate(global_model, test_loader)
        test_losses_over_rounds.append(test_loss)
        accuracy_over_rounds.append(test_acc)

        print(f"Round {round_idx+1} - Avg local train loss: {mean_train_loss:.4f}")
        print(f"Round {round_idx+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

    #########################
    # PLOTS
    #########################

    # 1) Client data distribution
    plt.figure(figsize=(10, 6))
    ind = np.arange(num_clients)
    width = 0.08
    for cls in range(num_classes):
        counts = [client_data_distribution[cid][cls] for cid in range(num_clients)]
        plt.bar(ind + cls*width, counts, width, label=f"Class {cls}")
    plt.xlabel("Client")
    plt.ylabel("Number of Samples")
    plt.title("Client Data Distribution per Class (IID or Dirichlet)")
    plt.xticks(ind + width*(num_classes-1)/2, [f"Client {i}" for i in range(num_clients)])
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "client_data_distribution.png"))
    plt.close()

    # 2) Plot accuracy: each client + global
    plt.figure(figsize=(9, 6))
    rounds = range(1, num_global_rounds+1)
    for client_idx in range(num_clients):
        plt.plot(
            rounds,
            accuracy_per_client_over_rounds[client_idx],
            marker='o',
            linestyle='-',
            label=f"Client {client_idx}"
        )
    plt.plot(
        rounds,
        accuracy_over_rounds,
        marker='s',
        linestyle='--',
        color='red',
        label="Global Model"
    )
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Client vs. Global Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "accuracy_clients_global.png"))
    plt.close()

    # 3a) Learning rate (linear scale)
    plt.figure(figsize=(8, 6))
    for client_idx in range(num_clients):
        plt.plot(
            rounds,
            lr_per_client_per_round[client_idx],
            marker='o',
            linestyle='-',
            label=f"Client {client_idx}"
        )
    plt.xlabel("Global Round")
    plt.ylabel("Learning Rate (SASS)")
    plt.title("Learning Rate per Client (Linear Scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "lr_linear.png"))
    plt.close()

    # 3b) Learning rate (log scale)
    plt.figure(figsize=(8, 6))
    for client_idx in range(num_clients):
        plt.plot(
            rounds,
            lr_per_client_per_round[client_idx],
            marker='o',
            linestyle='-',
            label=f"Client {client_idx}"
        )
    plt.yscale('log')
    plt.xlabel("Global Round")
    plt.ylabel("Learning Rate (Log Scale)")
    plt.title("Learning Rate per Client (Log Scale)")
    plt.legend()
    plt.grid(True, which='both')
    plt.savefig(os.path.join(save_path, "lr_log.png"))
    plt.close()

    # 4) Plot global test accuracy vs global rounds
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, accuracy_over_rounds, 'o-r', linestyle='--')
    plt.xlabel("Global Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Global Model Accuracy vs. Global Round")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "global_test_accuracy.png"))
    plt.close()

    # 5) Plot average local train loss vs global round
    plt.figure(figsize=(8, 6))
    plt.plot(rounds, train_losses_over_rounds, 'o-b')
    plt.xlabel("Global Round")
    plt.ylabel("Avg Local Train Loss")
    plt.title("Average Local Training Loss vs. Global Round")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "train_loss_vs_round.png"))
    plt.close()

    print("\nTraining completed. All plots saved in:", save_path)


if __name__ == "__main__":
    main()
