# python /home/local/ASURITE/yzeng88/fedSASS/SASS/test2_FedAvg.py --num_clients 1 --global_rounds 50 --local_epochs 1 --dist dirichlet --alpha2 0.2
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/test2_FedAvg.py --num_clients 3 --global_rounds 200 --local_epochs 1 --dist dirichlet --alpha2 0.2
# python /home/local/ASURITE/yzeng88/fedSASS/SASS/test_Feb3_fedAvg_case.py --num_clients 5 --global_rounds 20 --local_epochs 1 --dist case1

# python /home/local/ASURITE/yzeng88/fedSASS/SASS/fedAvg_case_experiment.py --num_clients 5 --global_rounds 20 --local_epochs 1 --dist case1
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

###############################################################################
# 1) PARTITIONING UTILS
###############################################################################

def simple_distribution_plot(num_clients, num_classes, distribution, filename):
    """
    A "Plot B" style distribution visualization using horizontal stacked bars.
    'distribution' is shape [num_clients, num_classes],
    i.e. distribution[cid][cls] = # elements for client cid of class cls.
    This function draws a horizontally stacked bar chart.
    """
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
        ax.barh(range(num_clients), dist_array[cls_idx],
                left=left_val, color=colors[cls_idx % len(colors)])

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


def case1_partition(train_dataset, num_clients, seed=42, output_path=None):
    """
    Partition strategy:
    - Client 0: 600 samples from class 0
    - Other clients: Evenly split remaining data (including remaining class 0 samples)
    """
    np.random.seed(seed)

    # Get all class 0 indices
    class0_indices = [idx for idx, (_, label) in enumerate(train_dataset) if label == 0]
    np.random.shuffle(class0_indices)

    # Client 0 gets first 600 class 0 samples (or all if <600)
    client0_indices = class0_indices[:600]
    remaining_indices = class0_indices[600:]  # Could be empty

    # Get indices for other classes
    other_indices = [idx for idx, (_, label) in enumerate(train_dataset) if label != 0]
    combined_remaining = remaining_indices + other_indices
    np.random.shuffle(combined_remaining)

    # Split remaining data among other clients
    num_remaining_clients = num_clients - 1
    if num_remaining_clients > 0:
        splits = np.array_split(combined_remaining, num_remaining_clients)
    else:
        splits = [combined_remaining]  # Handle case where num_clients=1

    # Create client datasets
    client_datasets = [Subset(train_dataset, client0_indices)]
    for split in splits:
        client_datasets.append(Subset(train_dataset, split))

    # Plot distribution
    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)

    return client_datasets

###############################################################################
# 2) MODEL
###############################################################################
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


###############################################################################
# 3) SASS OPTIMIZER
###############################################################################
class Sass(torch.optim.Optimizer):
    def __init__(self, params,
                 init_step_size=1.0, theta=0.2,
                 gamma_decr=0.7, gamma_incr=1.25,
                 alpha_max=10.0, eps_f=0.0):
        defaults = dict(
            init_step_size=init_step_size,
            theta=theta,
            gamma_decr=gamma_decr,
            gamma_incr=1.25,
            alpha_max=alpha_max,
            eps_f=0.0
        )
        super().__init__(params, defaults)
        self.state["step_size"] = init_step_size

    def step(self, closure):
        if closure is None:
            raise RuntimeError("SASS requires a closure that returns the forward loss.")
        loss = closure()
        loss.backward()

        step_size = self.state["step_size"]
        grad_norm_sq = 0.0

        for group in self.param_groups:
            params_current = [p.data.clone() for p in group["params"]]

            for p in group["params"]:
                if p.grad is not None:
                    grad_norm_sq += p.grad.data.norm()**2

            # Proposed update
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= step_size * p.grad.data

            # Evaluate new loss
            loss_next = closure()  # forward pass only
            lhs = loss_next
            rhs = loss - group["theta"] * step_size * grad_norm_sq + group["eps_f"]

            # Armijo check
            if lhs <= rhs:
                step_size = min(step_size * group["gamma_incr"], group["alpha_max"])
            else:
                step_size = step_size * group["gamma_decr"]
                for p, pcurr in zip(group["params"], params_current):
                    p.data.copy_(pcurr)

        self.state["step_size"] = step_size
        grad_norm = float(grad_norm_sq**0.5)
        self.state["grad_norm"] = grad_norm
        return loss


###############################################################################
# 4) LOCAL TRAINING + Logging Loss vs NN Passes
###############################################################################
def evaluate_loss(model, dataset, device="cpu", batch_size=1000):
    """
    Helper to compute average NLL test loss over 'dataset'.
    """
    model.eval()
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += F.nll_loss(output, target, reduction='sum').item()
    avg_loss = total_loss / len(dataset)
    model.cpu()
    return avg_loss


def local_train_sass(model, 
                     train_dataset, 
                     device, 
                     optimizer,
                     cid,                    # <-- ADDED: client id
                     nn_passes_counters,     # <-- ADDED: list tracking "nn passes" for each client
                     train_loss_vs_passes,   # <-- ADDED: store (pass, train_loss)
                     test_loss_vs_passes,    # <-- ADDED: store (pass, test_loss)
                     test_dataset,           # <-- to measure test loss after each epoch
                     local_epochs=1, 
                     batch_size=32):
    """
    Modified to log (1) training loss after each mini-batch and
    (2) test loss after each local epoch, vs. the "NN pass" index.
    """
    model.train()
    model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []
    epoch_lrs    = []
    all_grad_norms = []

    for _ in range(local_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            def closure():
                optimizer.zero_grad()
                output = model(data)
                return F.nll_loss(output, target)

            loss = optimizer.step(closure)
            pass_index = nn_passes_counters[cid]

            # -- Log the training loss vs. NN pass --
            train_loss_vs_passes[cid].append((pass_index, loss.item()))
            nn_passes_counters[cid] += 1

            current_lr = optimizer.state["step_size"]
            current_gn = optimizer.state["grad_norm"]
            all_grad_norms.append(current_gn)

            epoch_losses.append(loss.item())
            epoch_lrs.append(current_lr)

        # After each local epoch, measure test loss vs. passes
        test_l = evaluate_loss(model, test_dataset, device)
        test_loss_vs_passes[cid].append((nn_passes_counters[cid], test_l))

    model.cpu()
    avg_grad_norm = float(np.mean(all_grad_norms)) if all_grad_norms else 0.0

    # Return the *average* epoch losses and LRs for convenience
    if len(epoch_losses) == 0:
        mean_loss = 0.0
        mean_lr   = optimizer.state["step_size"]
    else:
        mean_loss = float(np.mean(epoch_losses))
        mean_lr   = float(np.mean(epoch_lrs))

    return model, [mean_loss], [mean_lr], avg_grad_norm


###############################################################################
# 5) FEDAVG AGGREGATOR
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
# 6) EVALUATION
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
# 7) MAIN + ARGPARSE
###############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=2, help="number of clients")
    parser.add_argument("--global_rounds", type=int, default=5, help="number of global rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="local batch size")
    parser.add_argument("--dist", type=str, default="iid",
                        choices=["iid", "class", "dirichlet","extreme","case1"], help="data distribution")
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
    main_folder = "A_experiment_output"
    os.makedirs(main_folder, exist_ok=True)

    out_dir = f"fedavg_{args.dist}_{args.num_clients}clients_{args.global_rounds}rounds"
    out_dir = os.path.join(main_folder, out_dir)  # subfolder
    os.makedirs(out_dir, exist_ok=True)

    # We include a dummy 'extreme' partition if needed; else skip
    def extreme_noniid_partition(train_dataset_, num_clients_, seed_=42, output_path_=None):
        np.random.seed(seed_)
        all_labels = [label for _, label in train_dataset_]
        unique_labels = list(set(all_labels))
        num_classes = len(unique_labels)
        if num_clients_ > num_classes:
            raise ValueError("For 'extreme' partition, #clients cannot exceed #classes.")
        np.random.shuffle(unique_labels)
        chosen_labels = unique_labels[:num_clients_]
        class_indices = {c: [] for c in chosen_labels}
        for idx, (_, label) in enumerate(train_dataset_):
            if label in chosen_labels:
                class_indices[label].append(idx)
        client_datasets_ = []
        for i in range(num_clients_):
            label_i = chosen_labels[i]
            subset_i = Subset(train_dataset_, class_indices[label_i])
            client_datasets_.append(subset_i)
        if output_path_:
            # For MNIST, we pass num_classes=10 for correct x-axis labeling
            plot_distribution(client_datasets_, num_clients_, 10, output_path_)
        return client_datasets_

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
    elif args.dist == "case1":
        client_datasets = case1_partition(train_dataset, args.num_clients,
                                          seed=seed,
                                          output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "extreme":
        client_datasets = extreme_noniid_partition(train_dataset, args.num_clients,
                                                   seed=seed,
                                                   output_path=os.path.join(out_dir, "data_dist.png"))
    else:
        raise ValueError("Unknown dist partition type!")

    # 3) Initialize global model
    global_model = SmallNet()

    # Create a SASS optimizer per client
    client_optimizers = []
    for cid in range(args.num_clients):
        opt = Sass(global_model.parameters(),
                   init_step_size=1.0,
                   theta=0.2,
                   gamma_decr=0.7,
                   gamma_incr=1.25,
                   alpha_max=10.0,
                   eps_f=0.0)
        client_optimizers.append(opt)

    # 4) Logs
    global_accuracies = []
    client_accuracies = [[] for _ in range(args.num_clients)]
    client_epoch_losses = [[] for _ in range(args.num_clients)]
    client_epoch_lrs    = [[] for _ in range(args.num_clients)]
    client_lr_final_per_round = [[] for _ in range(args.num_clients)]
    client_grad_norms_per_round = [[] for _ in range(args.num_clients)]

    # NEW: to track training & test loss vs. NN passes
    nn_passes_counters = [0]*args.num_clients
    train_loss_vs_passes = [[] for _ in range(args.num_clients)]
    test_loss_vs_passes  = [[] for _ in range(args.num_clients)]

    result_path = os.path.join(out_dir, "result.txt")
    server_acc_path = os.path.join(out_dir, "server_accuracy.txt")

    with open(result_path, "w") as f, open(server_acc_path, "w") as f_server:
        # Header lines
        f.write(f"{'GlobalRnd':<12}{'ClientID':<10}{'LocEpoch':<12}{'Accuracy':<12}"
                f"{'LR':<12}{'Loss':<12}{'GradNorm':<12}\n")
        f_server.write("GlobalRound\tGlobalAccuracy\n")

        # 5) Federated Rounds
        for rnd in range(args.global_rounds):
            print(f"\n--- Global Round {rnd+1}/{args.global_rounds} ---")
            local_models = []

            for cid in range(args.num_clients):
                print(f"[Client {cid}] local training ...")
                # Copy global model
                client_model = copy.deepcopy(global_model)

                # Link the SASS optimizer to this client's local model
                client_optimizers[cid].param_groups[0]["params"] = list(client_model.parameters())

                updated_model, epoch_losses, epoch_lrs, avg_grad_norm = local_train_sass(
                    model=client_model,
                    train_dataset=client_datasets[cid],
                    device=device,
                    optimizer=client_optimizers[cid],
                    cid=cid,
                    nn_passes_counters=nn_passes_counters,
                    train_loss_vs_passes=train_loss_vs_passes,
                    test_loss_vs_passes=test_loss_vs_passes,
                    test_dataset=test_dataset,
                    local_epochs=args.local_epochs,
                    batch_size=args.batch_size
                )
                local_models.append(updated_model)

                # Evaluate updated model
                cacc = evaluate(updated_model, test_dataset, device=device)
                client_accuracies[cid].append(cacc)
                print(f"  Client {cid} accuracy after local update: {cacc:.2f}%")

                # Logging
                client_epoch_losses[cid].extend(epoch_losses)
                client_epoch_lrs[cid].extend(epoch_lrs)
                final_lr_this_round = epoch_lrs[-1]
                final_loss_this_round = epoch_losses[-1]
                client_lr_final_per_round[cid].append(final_lr_this_round)
                client_grad_norms_per_round[cid].append(avg_grad_norm)

                f.write(f"{rnd+1:<12}{cid:<10}{args.local_epochs:<12}{cacc:<12.4f}"
                        f"{final_lr_this_round:<12.6f}{final_loss_this_round:<12.6f}{avg_grad_norm:<12.6f}\n")

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
    plt.title("Client & Global Accuracy vs. Global Round (FedAvg + SASS)")
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_clients_and_global.png"))
    plt.close()

    # (B) Final LR per Client per Global Round
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_lr_final_per_round[cid], linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client per Global Round (FedAvg + SASS)")
    plt.xlabel("Global Round")
    plt.ylabel("Final LR in Local Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_round.png"))
    plt.close()

    # (B2) Log10(LR) per Client
    plt.figure()
    for cid in range(args.num_clients):
        lr_log = [np.log10(lr) for lr in client_lr_final_per_round[cid]]
        plt.plot(rounds_range, lr_log, linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.title("Log10(LR) per Client per Global Round (FedAvg + SASS)")
    plt.xlabel("Global Round")
    plt.ylabel("log10(Final LR)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_round_log.png"))
    plt.close()

    # (C) Per-client LR across local epochs
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(range(1, len(client_epoch_lrs[cid]) + 1),
                 client_epoch_lrs[cid], linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client (Local Epoch Index)")
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("LR (Step Size)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "learning_rate_per_client_local_epochs.png"))
    plt.close()

    # (D) Per-client Training Loss
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

    # (E) Gradient Norm per Client per Global Round
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_grad_norms_per_round[cid], linestyle="-",
                 linewidth=2, label=f"Client {cid}")
    plt.title("Gradient Norm per Client per Global Round (FedAvg + SASS)")
    plt.xlabel("Global Round")
    plt.ylabel("Avg Gradient Norm (Local Training)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "grad_norm_per_client_per_global_round.png"))
    plt.close()

    # --------------------------------------------------------
    # NEW PLOTS: Training Loss & Test Loss vs NN Passes (per client)
    # --------------------------------------------------------
    # (F) Training Loss vs. NN Passes per client (subplots)
    fig, axes = plt.subplots(nrows=1, ncols=args.num_clients, figsize=(6*args.num_clients, 5))
    if args.num_clients == 1:
        axes = [axes]  # If only 1 client, 'axes' is single object

    for cid in range(args.num_clients):
        if len(train_loss_vs_passes[cid]) > 0:
            passes, losses = zip(*train_loss_vs_passes[cid])
            axes[cid].plot(passes, losses, color='blue', linewidth=1)
            axes[cid].set_title(f"Client {cid} - Train Loss vs NN Passes")
            axes[cid].set_xlabel("NN Passes")
            axes[cid].set_ylabel("Train Loss")
            axes[cid].grid(True)
        else:
            axes[cid].set_title(f"Client {cid} - No Train Data")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_vs_nn_passes_per_client.png"))
    plt.close()

    # (G) Test Loss vs. NN Passes per client (subplots)
    fig, axes = plt.subplots(nrows=1, ncols=args.num_clients, figsize=(6*args.num_clients, 5))
    if args.num_clients == 1:
        axes = [axes]

    for cid in range(args.num_clients):
        if len(test_loss_vs_passes[cid]) > 0:
            passes, losses = zip(*test_loss_vs_passes[cid])
            axes[cid].plot(passes, losses, color='red', marker='o', linewidth=1)
            axes[cid].set_title(f"Client {cid} - Test Loss vs NN Passes")
            axes[cid].set_xlabel("NN Passes")
            axes[cid].set_ylabel("Test Loss")
            axes[cid].grid(True)
        else:
            axes[cid].set_title(f"Client {cid} - No Test Data")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_loss_vs_nn_passes_per_client.png"))
    plt.close()
    # -----------------------------------------------------
