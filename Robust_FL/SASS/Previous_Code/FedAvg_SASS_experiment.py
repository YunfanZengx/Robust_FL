
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/FedAvg_SASS_experiment.py --num_clients 1 --num_global_rounds 50 --local_epochs 1 --partition dirichlet --alpha 0.2
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/FedAvg_SASS_experiment.py --num_clients 3 --num_global_rounds 50 --local_epochs 1 --partition iid
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/FedAvg_SASS_experiment.py --num_clients 10 --num_global_rounds 50 --local_epochs 2 --partition dirichlet --alpha 0.2
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/FedAvg_SASS_experiment.py --num_clients 10 --num_global_rounds 50 --local_epochs 2 --partition iid

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Import your SASS optimizer
from sass import Sass


# 1) DATA PARTITIONING
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
            end = dataset_size
        subset_indices = all_indices[start:end]
        subset_ds = Subset(full_dataset, subset_indices)
        client_datasets.append(subset_ds)
        start = end
    return client_datasets

def dirichlet_noniid_partition(full_dataset, num_clients, alpha=0.5, seed=42):
    np.random.seed(seed)
    labels = [sample[1] for sample in full_dataset]
    num_classes = len(set(labels))
    class_indices = {cls: [] for cls in range(num_classes)}

    for idx, lab in enumerate(labels):
        class_indices[lab].append(idx)
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    client_class_proportions = np.random.dirichlet([alpha]*num_clients, num_classes)
    client_indices = [[] for _ in range(num_clients)]
    for cls_idx in range(num_classes):
        cindices = class_indices[cls_idx]
        if len(cindices) == 0:
            continue
        num_cls_samples = len(cindices)
        proportions = client_class_proportions[cls_idx]
        class_split = (proportions * num_cls_samples).astype(int)

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

# 2) MODEL

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
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            return loss

        loss = optimizer.step(closure)
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)
    accuracy = 100.0 * correct / len(dataloader.dataset)
    return test_loss, accuracy

def compute_grad_norm(model):
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq += p.grad.data.norm(2).item()**2
    return grad_norm_sq**0.5

def Federated_Average(list_of_models):
    base_model = list_of_models[0]
    base_params = base_model.state_dict()
    for key in base_params.keys():
        for i in range(1, len(list_of_models)):
            base_params[key] += list_of_models[i].state_dict()[key]
        base_params[key] = base_params[key] / len(list_of_models)
    return base_params

########################################################################
# 4) SIMPLE DISTRIBUTION PLOT
########################################################################
def simple_distribution_plot(num_clients, num_classes, distribution, filename):
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
    ax.barh(range(num_clients), dist_array[0], color=colors[0])
    for i in range(1, num_classes):
        left_val = np.sum(dist_array[:i], axis=0)
        ax.barh(range(num_clients), dist_array[i], left=left_val, color=colors[i % len(colors)])

    ax.set_ylabel("Client")
    ax.set_xlabel("Number of Elements")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(filename)
    plt.close()

########################################################################
# 5) MAIN with ARGPARSE + dynamic results folder
########################################################################
def main():
    import argparse

    parser = argparse.ArgumentParser(description="SASS Federated Training with Argparse")
    parser.add_argument('--num_clients', type=int, default=5, help="Number of clients")
    parser.add_argument('--num_global_rounds', type=int, default=50, help="Number of global rounds")
    parser.add_argument('--local_epochs', type=int, default=2, help="Local epochs per client each round")
    parser.add_argument('--partition', type=str, default='iid',
                        choices=['iid', 'dirichlet'],
                        help="Data partition type (iid or dirichlet)")
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Dirichlet alpha (only used if partition=dirichlet)")
    args = parser.parse_args()

    print("==== Experiment Settings ====")
    print(f"Clients          = {args.num_clients}")
    print(f"Global rounds    = {args.num_global_rounds}")
    print(f"Local epochs     = {args.local_epochs}")
    print(f"Partition        = {args.partition}")
    if args.partition == 'dirichlet':
        print(f"Alpha            = {args.alpha}")
    print("============================\n")

    # Decide folder name based on parameters
    base_folder = "./FedAvg_SASS_results"
    os.makedirs(base_folder, exist_ok=True)

    exp_name = f"FedAvg_{args.num_clients}clients_{args.num_global_rounds}global_{args.local_epochs}local_{args.partition}"
    if args.partition == 'dirichlet':
        exp_name += f"-alpha{args.alpha}"

    save_path = os.path.join(base_folder, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Seeds
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Federated parameters
    num_clients = args.num_clients
    num_global_rounds = args.num_global_rounds
    local_epochs = args.local_epochs

    # Data
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

    # Partition
    if args.partition == 'iid':
        client_datasets = iid_partition(full_train_dataset, num_clients, seed=seed)
    else:
        client_datasets = dirichlet_noniid_partition(full_train_dataset, num_clients, args.alpha, seed=seed)

    # Model
    global_model = SmallNet().to(device)
    global_model_weights = copy.deepcopy(global_model.state_dict())

    # Stats
    client_results_buffer = []
    server_results_buffer = []

    accuracy_per_client_over_rounds = [[] for _ in range(num_clients)]
    lr_per_client_per_round = [[] for _ in range(num_clients)]
    train_loss_per_client_over_rounds = [[] for _ in range(num_clients)]
    grad_norm_per_client_over_rounds = [[] for _ in range(num_clients)]
    global_accuracy = []

    # data distribution
    num_classes = 10
    client_data_distribution = []
    for cidx in range(num_clients):
        labels = [full_train_dataset.targets[idx].item() for idx in client_datasets[cidx].indices]
        class_counts = [labels.count(c) for c in range(num_classes)]
        client_data_distribution.append(class_counts)

    # Federated Training
    for round_idx in range(num_global_rounds):
        print(f"\n--- Global Round {round_idx+1}/{num_global_rounds} ---")
        local_models = []

        for client_idx in range(num_clients):
            local_model = SmallNet().to(device)
            local_model.load_state_dict(copy.deepcopy(global_model_weights))

            local_loader = DataLoader(client_datasets[client_idx], batch_size=64, shuffle=True)
            n_batches_per_epoch = len(local_loader)
            local_optimizer = Sass(local_model.parameters(), n_batches_per_epoch=n_batches_per_epoch)

            for ep in range(local_epochs):
                avg_loss = train_one_epoch(local_model, local_loader, local_optimizer, device)
                # Evaluate
                client_eval_loader = DataLoader(client_datasets[client_idx], batch_size=500, shuffle=False)
                c_loss, c_acc = evaluate(local_model, client_eval_loader, device)

                if local_optimizer.state['step_size_vs_nn_passes']:
                    last_step_size = local_optimizer.state['step_size_vs_nn_passes'][-1][1]
                else:
                    last_step_size = float('nan')

                grad_norm = compute_grad_norm(local_model)

                client_results_buffer.append((
                    round_idx+1, client_idx, ep+1, c_acc, last_step_size, avg_loss, grad_norm
                ))

            train_loss_per_client_over_rounds[client_idx].append(avg_loss)
            grad_norm_per_client_over_rounds[client_idx].append(grad_norm)
            accuracy_per_client_over_rounds[client_idx].append(c_acc)
            lr_per_client_per_round[client_idx].append(last_step_size)

            local_models.append(local_model)

        if num_clients > 1:
            new_weights = Federated_Average(local_models)
            global_model.load_state_dict(new_weights)
            global_model_weights = copy.deepcopy(new_weights)
        else:
            global_model.load_state_dict(local_models[0].state_dict())
            global_model_weights = copy.deepcopy(local_models[0].state_dict())

        _, test_acc = evaluate(global_model, test_loader, device)
        global_accuracy.append(test_acc)
        server_results_buffer.append((round_idx+1, test_acc))
        print(f"Round {round_idx+1} - Global Accuracy: {test_acc:.2f}%")

    # Write result files
    client_result_path = os.path.join(save_path, "client_result.txt")
    with open(client_result_path, "w") as f:
        f.write(
            f"{'GlobalRnd':>8} {'ClientID':>8} {'LocEpoch':>8} "
            f"{'Accuracy':>10} {'LR':>10} {'Loss':>10} {'GradNorm':>10}\n"
        )
        for row in client_results_buffer:
            rnd, cid, lep, acc, lrval, lossv, gnorm = row
            f.write(
                f"{rnd:>8} {cid:>8} {lep:>8} {acc:>10.4f} "
                f"{lrval:>10.6f} {lossv:>10.6f} {gnorm:>10.6f}\n"
            )

    server_result_path = os.path.join(save_path, "server_result.txt")
    with open(server_result_path, "w") as f:
        f.write(f"{'GlobalRnd':>8} {'Accuracy':>10}\n")
        for rnd, acc in server_results_buffer:
            f.write(f"{rnd:>8} {acc:>10.4f}\n")

    # Plot data distribution
    dist_fig_path = os.path.join(save_path, "client_data_distribution.png")
    simple_distribution_plot(num_clients, num_classes, client_data_distribution, dist_fig_path)

    # Additional plots
    rounds_x = range(1, num_global_rounds + 1)

    # (1) accuracy: clients + global
    plt.figure(figsize=(10, 7))
    for cidx in range(num_clients):
        plt.plot(
            rounds_x,
            accuracy_per_client_over_rounds[cidx],
            marker='o',
            linestyle='-',
            label=f"Client {cidx}"
        )
    plt.plot(
        rounds_x,
        global_accuracy,
        marker='s',
        linestyle='--',
        color='red',
        label="Global Model"
    )
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy: Clients vs. Global Model")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "accuracy_clients_global.png"))
    plt.close()

    # (2) LR linear
    plt.figure(figsize=(8, 6))
    for cidx in range(num_clients):
        plt.plot(
            rounds_x,
            lr_per_client_per_round[cidx],
            marker='o',
            linestyle='-',
            label=f"Client {cidx}"
        )
    plt.xlabel("Global Round")
    plt.ylabel("Learning Rate (SASS)")
    plt.title("Learning Rate per Client (Linear Scale)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "learning_rate_linear.png"))
    plt.close()

    # (3) LR log
    plt.figure(figsize=(8, 6))
    for cidx in range(num_clients):
        plt.plot(
            rounds_x,
            lr_per_client_per_round[cidx],
            marker='o',
            linestyle='-',
            label=f"Client {cidx}"
        )
    plt.yscale('log')
    plt.xlabel("Global Round")
    plt.ylabel("Learning Rate (log scale)")
    plt.title("Learning Rate per Client (Log Scale)")
    plt.legend()
    plt.grid(True, which='both')
    plt.savefig(os.path.join(save_path, "learning_rate_log.png"))
    plt.close()

    # (4) Per client training loss
    plt.figure(figsize=(8, 6))
    for cidx in range(num_clients):
        plt.plot(
            rounds_x,
            train_loss_per_client_over_rounds[cidx],
            marker='o',
            linestyle='-',
            label=f"Client {cidx}"
        )
    plt.xlabel("Global Round")
    plt.ylabel("Training Loss")
    plt.title("Per-Client Training Loss vs. Round")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "train_loss_per_client.png"))
    plt.close()


    print("\nAll done! Results saved in:", save_path)

if __name__ == "__main__":
    main()


# python /home/local/ASURITE/yzeng88/fedSASS/SASS/FedAvg_SASS_experiment.py --num_clients 5 --num_global_rounds 50 --local_epochs 2 --partition dirichlet --alpha 0.2