import os
import copy
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/test2_FedAvg_SGD_extreme.py --num_clients 10 --global_rounds 50 --local_epochs 10 --dist extreme



def simple_distribution_plot(num_clients, num_classes, distribution, filename):
    """
    A "Plot B" style distribution visualization using horizontal stacked bars.
    'distribution' is shape [num_clients, num_classes],
    i.e. distribution[cid][cls] = # elements for client cid of class cls.
    This function draws a horizontally stacked bar chart.
    """
    import matplotlib.pyplot as plt

    dist_array = np.array(distribution).T  # shape [num_classes, num_clients]

    fig, ax = plt.subplots(figsize=(20, num_clients/2 + 3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    colors = [
        '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', 
        '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', 
        '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', 
        '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
    ]

    # Plot the first class distribution as the base bar
    ax.barh(range(num_clients), dist_array[0], color=colors[0])
    # Stack the remaining classes
    for cls_idx in range(1, num_classes):
        left_val = np.sum(dist_array[:cls_idx], axis=0)
        ax.barh(range(num_clients), dist_array[cls_idx], left=left_val,
                color=colors[cls_idx % len(colors)])

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.savefig(filename)
    plt.close()


def plot_distribution(client_datasets, num_clients, num_classes, output_path):
    # --- Original vertical bar plot ---
    class_counts = [torch.zeros(num_classes, dtype=torch.int32) for _ in range(num_clients)]
    for i, dataset in enumerate(client_datasets):
        labels = [label for _, label in dataset]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[i][cls] = count

    plt.figure(figsize=(10, 6))
    for i in range(num_clients):
        plt.bar(np.arange(num_classes) + i * 0.1, 
                class_counts[i], width=0.1, label=f'Client {i+1}')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution Across Clients (Plot A)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

    # --- Added: "Plot B" horizontal stacked bar ---
    distribution_list = [cc.tolist() for cc in class_counts]  # shape [num_clients, num_classes]
    output_path_b = output_path.replace(".png", "_B.png")
    simple_distribution_plot(num_clients, num_classes, distribution_list, output_path_b)


def iid_partition(train_dataset, num_clients, output_path=None):
    num_samples = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_datasets = []
    for i in range(num_clients):
        start = i * num_samples
        end   = (i + 1) * num_samples
        subset_indices = indices[start:end]
        client_datasets.append(Subset(train_dataset, subset_indices))

    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    return client_datasets


def class_based_noniid_partition(train_dataset, num_clients, seed=42, output_path=None):
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {c: [] for c in labels}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    for c in labels:
        splits = np.array_split(class_indices[c], num_clients)
        for i in range(num_clients):
            client_indices[i].extend(splits[i])

    client_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]
    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    return client_datasets


def dirichlet_noniid_partition(train_dataset, num_clients, alpha2=0.5, seed=42, output_path=None):
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {c: [] for c in labels}
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    # Dirichlet draws
    client_class_proportions = np.random.dirichlet([alpha2]*num_clients, num_classes)

    client_indices = [[] for _ in range(num_clients)]
    for c in labels:
        cinds = class_indices[c]
        num_samples_per_client = np.maximum(
            1, (client_class_proportions[c] * len(cinds)).astype(int)
        )
        cum_samples = np.cumsum(num_samples_per_client)
        start_idx = 0
        for i in range(num_clients):
            end_idx = min(cum_samples[i], len(cinds))
            if start_idx < end_idx:
                client_indices[i].extend(cinds[start_idx:end_idx])
            start_idx = end_idx

    # Ensure no empty subset
    for i, idxs in enumerate(client_indices):
        if len(idxs) == 0:
            max_client_idx = np.argmax([len(d) for d in client_indices])
            client_indices[i].append(client_indices[max_client_idx].pop())

    client_datasets = [Subset(train_dataset, idxs) for idxs in client_indices]
    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    return client_datasets

def extreme_noniid_partition(train_dataset, num_clients, seed=42, output_path=None):
    """
    Assign exactly ONE distinct class to each client (client i only has samples from class_i).
    If num_clients > num_classes, this function currently raises an error.
    """
    np.random.seed(seed)
    
    # Gather labels
    all_labels = [label for _, label in train_dataset]
    unique_labels = list(set(all_labels))
    num_classes = len(unique_labels)
    
    # Safety check
    if num_clients > num_classes:
        raise ValueError(f"For 'extreme' partition, the number of clients ({num_clients}) "
                         f"cannot exceed the number of classes ({num_classes}).")
    
    # Randomly shuffle the available classes and pick as many as clients
    np.random.shuffle(unique_labels)
    chosen_labels = unique_labels[:num_clients]

    # Prepare a dict to collect indices by chosen label
    class_indices = {c: [] for c in chosen_labels}
    for idx, (_, label) in enumerate(train_dataset):
        if label in chosen_labels:
            class_indices[label].append(idx)

    # Create a subset for each client
    client_datasets = []
    for i in range(num_clients):
        label_i = chosen_labels[i]
        subset_i = Subset(train_dataset, class_indices[label_i])
        client_datasets.append(subset_i)

    # Optionally plot the distribution
    if output_path:
        # IMPORTANT: pass num_classes=10 (for MNIST) so the plot has the correct x-axis.
        # However, each client will have count=0 for the classes it does not own.
        plot_distribution(client_datasets, num_clients, 10, output_path)

    return client_datasets

###############################################################################
# 2) MODEL
###############################################################################
# class SmallNet(nn.Module):
#     def __init__(self):
#         super(SmallNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         return F.log_softmax(self.fc2(x), dim=1)

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

###############################################################################
# 3) LOCAL TRAINING WITH SGD
###############################################################################
def local_train_sgd(model, train_dataset, device, local_epochs=1, batch_size=64, lr=0.01):
    """
    Local training using standard SGD with fixed learning rate.
    """
    model.train()
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr=lr)

    epoch_losses = []

    for _ in range(local_epochs):
        total_loss = 0.0
        n_batches = len(train_loader)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_losses.append(total_loss / n_batches)

    model.cpu()
    return model, epoch_losses


###############################################################################
# 4) FEDAVG AGGREGATOR
###############################################################################
def fedavg_aggregate(global_model, local_models):
    if len(local_models) == 1:
        # trivial
        global_model.load_state_dict(local_models[0].state_dict())
        return global_model

    global_dict = global_model.state_dict()
    num_clients = len(local_models)
    for key in global_dict.keys():
        stack = torch.stack([m.state_dict()[key] for m in local_models], dim=0)
        global_dict[key] = torch.mean(stack, dim=0)

    global_model.load_state_dict(global_dict)
    return global_model


###############################################################################
# 5) EVALUATION
###############################################################################
def evaluate(model, test_dataset, device="cpu", batch_size=1000):
    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total   += target.size(0)
    acc = 100.0 * correct / total
    model.cpu()
    return acc


###############################################################################
# 6) MAIN + ARGPARSE
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=2, help="number of clients")
    parser.add_argument("--global_rounds", type=int, default=5, help="number of global rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch_size", type=int, default=64, help="local batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="local learning rate for SGD")
    parser.add_argument("--dist", type=str, default="iid",
                        choices=["iid", "class", "dirichlet","extreme"], 
                        help="data distribution")
    parser.add_argument("--alpha2", type=float, default=0.5,
                        help="Dirichlet alpha2 parameter (only relevant if dist=dirichlet)")
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load data
    data_dir = "./mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # 2) Partition
    main_folder = "Jan29_output_SGD"
    os.makedirs(main_folder, exist_ok=True)

    out_dir = f"fedavg_sameCNN_{args.dist}_{args.num_clients}clients_{args.global_rounds}rounds_SGD_momentum0.9"
    out_dir = os.path.join(main_folder, out_dir)  # subfolder
    os.makedirs(out_dir, exist_ok=True)

    if args.dist == "iid":
        client_datasets = iid_partition(train_dataset, args.num_clients,
                                        output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "class":
        client_datasets = class_based_noniid_partition(train_dataset, args.num_clients,
                                                       seed=seed,
                                                       output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "dirichlet":
        client_datasets = dirichlet_noniid_partition(train_dataset, args.num_clients,
                                                     alpha2=args.alpha2,
                                                     seed=seed,
                                                     output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "extreme":
        client_datasets = extreme_noniid_partition(train_dataset, args.num_clients,
                                                   seed=seed,
                                                   output_path=os.path.join(out_dir, "data_dist.png"))
    
    # 3) Initialize global model
    global_model = SmallNet()

    # 4) Logs
    global_accuracies = []
    client_accuracies = [[] for _ in range(args.num_clients)]
    client_epoch_losses = [[] for _ in range(args.num_clients)]

    result_path = os.path.join(out_dir, "result.txt")
    server_acc_path = os.path.join(out_dir, "server_accuracy.txt")

    with open(result_path, "w") as f, open(server_acc_path, "w") as f_server:
        # Header lines
        f.write(f"{'GlobalRnd':<12}{'ClientID':<10}{'LocEpoch':<12}{'Accuracy':<12}{'Loss':<12}\n")
        f_server.write("GlobalRound\tGlobalAccuracy\n")

        # 5) Federated Rounds
        for rnd in range(args.global_rounds):
            print(f"\n--- Global Round {rnd+1}/{args.global_rounds} ---")
            local_models = []

            for cid in range(args.num_clients):
                print(f"[Client {cid}] local training with SGD...")
                # Copy global model
                client_model = copy.deepcopy(global_model)

                # Perform local training using standard SGD
                updated_model, epoch_losses = local_train_sgd(
                    client_model,
                    client_datasets[cid],
                    device,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size,
                    lr=args.lr
                )
                local_models.append(updated_model)

                # Evaluate updated model
                cacc = evaluate(updated_model, test_dataset, device=device)
                client_accuracies[cid].append(cacc)
                print(f"  Client {cid} accuracy after local update: {cacc:.2f}%")

                # Logging
                client_epoch_losses[cid].extend(epoch_losses)
                final_loss_this_round = epoch_losses[-1]
                f.write(f"{rnd+1:<12}{cid:<10}{args.local_epochs:<12}{cacc:<12.4f}{final_loss_this_round:<12.6f}\n")

            # FedAvg aggregator
            if args.num_clients > 1:
                global_model = fedavg_aggregate(global_model, local_models)
            else:
                global_model.load_state_dict(local_models[0].state_dict())

            # Evaluate global model
            global_acc = evaluate(global_model, test_dataset, device=device)
            global_accuracies.append(global_acc)
            print(f"Global accuracy after round {rnd+1}: {global_acc:.2f}%")

            # Write server accuracy
            f_server.write(f"{rnd+1}\t{global_acc:.4f}\n")

    ########################################################################
    # 6) PLOTS
    ########################################################################
    rounds_range = range(1, args.global_rounds + 1)

    # (A) Accuracy vs. Global Round
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_accuracies[cid], linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.plot(rounds_range, global_accuracies, marker='x',
             linewidth=2, linestyle="--", color="red", label="Global Acc")
    plt.title("Client & Global Accuracy vs. Global Round (FedAvg + SGD)")
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_clients_and_global.png"))
    plt.close()

    # (B) Per-client Training Loss
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(range(1, len(client_epoch_losses[cid]) + 1),
                 client_epoch_losses[cid], linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.title("Training Loss per Client (Local Epoch Index)")
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_loss_per_client.png"))
    plt.close()

    print(f"\nPlots saved in folder: '{out_dir}'")
    print(f"Client & server metrics saved to: '{result_path}', '{server_acc_path}'")


if __name__ == "__main__":
    main()
