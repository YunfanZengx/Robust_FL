import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

################################################################################
# 1) PARTITION CODE (Includes Plot of Distribution)
################################################################################
def plot_distribution(client_datasets, num_clients, num_classes, output_path):
    """Visualize the distribution of classes across clients."""
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    class_counts = [torch.zeros(num_classes) for _ in range(num_clients)]
    
    for i, dataset in enumerate(client_datasets):
        labels = [label for _, label in dataset]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[i][cls] = count

    plt.figure(figsize=(10, 6))
    for i in range(num_clients):
        plt.bar(np.arange(num_classes) + i * 0.1, class_counts[i], width=0.1, label=f'Client {i+1}')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution Across Clients")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def iid_partition(train_dataset, num_clients, output_path=None):
    """ IID partition: equally partition the dataset across clients. """
    num_samples = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_datasets = [
        Subset(train_dataset, indices[i * num_samples : (i + 1) * num_samples]) 
        for i in range(num_clients)
    ]

    # Plot distribution if desired
    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    
    return client_datasets

def class_based_noniid_partition(train_dataset, num_clients, seed=42, output_path=None):
    """Class-based Non-IID: each client gets a random subset of classes."""
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {i: [] for i in labels}
    
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    
    client_datasets_indices = [[] for _ in range(num_clients)]
    for cls in labels:
        split_indices = np.array_split(class_indices[cls], num_clients)
        for i in range(num_clients):
            client_datasets_indices[i].extend(split_indices[i])
    
    client_datasets = [Subset(train_dataset, client_datasets_indices[i]) for i in range(num_clients)]

    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    
    return client_datasets

def dirichlet_noniid_partition(train_dataset, num_clients, alpha1=8, alpha2=0.5, seed=42, output_path=None):
    """
    Dirichlet-based Non-IID partition.
    alpha2 controls class distribution skew. alpha1 is not used here but kept for signature compatibility.
    """
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {i: [] for i in labels}
    
    # Split dataset by class
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    # Shuffle each class
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # Dirichlet class proportions
    client_class_proportions = np.random.dirichlet([alpha2] * num_clients, num_classes)
    
    # Allocate
    client_datasets_indices = [[] for _ in range(num_clients)]
    for cls in labels:
        cindices = class_indices[cls]
        num_samples_per_client = np.maximum(
            1, (client_class_proportions[cls] * len(cindices)).astype(int)
        )
        cumulative_samples = np.cumsum(num_samples_per_client)
        
        start_idx = 0
        for i in range(num_clients):
            end_idx = min(cumulative_samples[i], len(cindices))
            if start_idx < end_idx:
                client_datasets_indices[i].extend(cindices[start_idx:end_idx])
            start_idx = end_idx

    # Ensure no client is empty
    for i, idxs in enumerate(client_datasets_indices):
        if len(idxs) == 0:
            max_client_idx = np.argmax([len(d) for d in client_datasets_indices])
            client_datasets_indices[i].append(client_datasets_indices[max_client_idx].pop())

    client_datasets = [Subset(train_dataset, idxs) for idxs in client_datasets_indices]

    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    
    return client_datasets

################################################################################
# 2) CNN MODEL FOR MNIST
################################################################################
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
        x = x.view(-1, 320)  # Flatten
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)

################################################################################
# 3) SASS OPTIMIZER
################################################################################
class Sass(torch.optim.Optimizer):
    """
    Minimal SASS (Stochastic Armijo Step Size) implementation in PyTorch.
    """
    def __init__(self, params, init_step_size=1.0, theta=0.2,
                 gamma_decr=0.7, gamma_incr=1.25, alpha_max=10.0, eps_f=1e-6):
        defaults = dict(
            init_step_size=init_step_size,
            theta=theta,
            gamma_decr=gamma_decr,
            gamma_incr=gamma_incr,
            alpha_max=alpha_max,
            eps_f=eps_f
        )
        super().__init__(params, defaults)
        self.state["step_size"] = init_step_size

    def step(self, closure):
        if closure is None:
            raise RuntimeError("SASS requires a closure.")
        
        # Evaluate initial loss
        loss = closure()
        loss.backward()

        step_size = self.state["step_size"]
        grad_norm_sq = 0.0

        for group in self.param_groups:
            # Save current params
            params_current = [p.data.clone() for p in group["params"]]

            # Compute grad norm^2
            for p in group["params"]:
                if p.grad is not None:
                    # Convert from GPU to CPU .item() so it's a float
                    grad_norm_sq += p.grad.data.norm().cpu().item() ** 2

            # Proposed step
            for p in group["params"]:
                if p.grad is not None:
                    p.data -= step_size * p.grad.data

            # Evaluate new loss
            loss_next = closure()  # no backward needed

            # Armijo condition
            lhs = loss_next
            rhs = loss - group["theta"] * step_size * grad_norm_sq + group["eps_f"]

            if lhs <= rhs:
                step_size = min(step_size * group["gamma_incr"], group["alpha_max"])
            else:
                step_size = step_size * group["gamma_decr"]
                for p, pcurr in zip(group["params"], params_current):
                    p.data.copy_(pcurr)

        # Store step_size back in self.state
        self.state["step_size"] = step_size
        
        # Store grad norm
        grad_norm = (grad_norm_sq ** 0.5)
        self.state["grad_norm"] = float(grad_norm)  # ensure it's a Python float

        return loss

################################################################################
# 4) LOCAL TRAINING (LOGS PER EPOCH LOSS + LR + GRAD NORM)
################################################################################
def local_train_sass(model, train_dataset, device, local_epochs=1, batch_size=32,
                     init_lr=1.0):
    """
    Returns:
      - updated model
      - epoch_losses: list of average training losses, length = local_epochs
      - epoch_lrs:    list of average step sizes, length = local_epochs
      - avg_grad_norm: the average gradient norm (across all mini-batches & epochs)
    """
    model.train()
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Sass(
        model.parameters(),
        init_step_size=init_lr,
        theta=0.2,
        gamma_decr=0.7,
        gamma_incr=1.25,
        alpha_max=10.0,
        eps_f=1e-6
    )

    epoch_losses = []
    epoch_lrs = []
    all_grad_norms = []  # track gradient norm for each mini-batch

    for _ in range(local_epochs):
        total_loss = 0.0
        total_lr = 0.0
        n_batches = len(train_loader)

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            def closure():
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                return loss

            loss = optimizer.step(closure)
            # print t
            # Current SASS step size:
            current_lr = optimizer.state["step_size"]
            current_grad_norm = optimizer.state["grad_norm"]

            all_grad_norms.append(current_grad_norm)
            total_loss += loss.item()
            total_lr += current_lr

        # Average for this epoch
        epoch_losses.append(total_loss / n_batches)
        epoch_lrs.append(total_lr / n_batches)

    model.cpu()
    avg_grad_norm = float(np.mean(all_grad_norms))

    return model, epoch_losses, epoch_lrs, avg_grad_norm

################################################################################
# GEOMETRIC MEDIAN AGGREGATOR (Weiszfeld)
################################################################################
def get_model_vector(model):
    """
    Flatten all parameters of a PyTorch model into a single 1-D numpy array.
    """
    with torch.no_grad():
        return np.concatenate([p.data.cpu().numpy().ravel() for p in model.parameters()])

def set_model_from_vector(model, vector):
    """
    Load parameters from a 1-D numpy array into a PyTorch model (in-place).
    """
    offset = 0
    for p in model.parameters():
        num_params = p.numel()
        new_vals = vector[offset : offset + num_params].reshape(p.shape)
        with torch.no_grad():
            p.data = torch.from_numpy(new_vals).type_as(p.data)
        offset += num_params

def weiszfeld_geometric_median(models, tol=1e-4, max_iter=100):
    """
    Compute geometric median (Weiszfeld's algorithm) of multiple PyTorch models.
    Returns a 1-D numpy array (the median vector).
    """
    vectors = [get_model_vector(m) for m in models]
    median = np.mean(vectors, axis=0)  # initial guess

    for _ in range(max_iter):
        distances = np.array([np.linalg.norm(v - median) for v in vectors])
        distances = np.clip(distances, 1e-12, None)  # avoid division by zero
        weights = 1.0 / distances
        new_median = np.sum([w * v for w, v in zip(weights, vectors)], axis=0) / np.sum(weights)

        if np.linalg.norm(new_median - median) < tol:
            median = new_median
            break
        median = new_median

    return median

################################################################################
# 6) EVALUATION (ACCURACY ON TEST)
################################################################################
def evaluate(model, test_dataset, device="cpu", batch_size=1000):
    model.eval()
    model.to(device)
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total
    model.cpu()
    return acc

################################################################################
# 7) MAIN FEDERATED LOOP
################################################################################
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load MNIST data
    data_dir = './mnist_data'
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # 2) Partition among clients
    num_clients = 1
    out_dir = "FedGM_MNIST_Extremenoniid_Dirichlet[80,0.1]_1_Client__50_epoch"
    #out_dir = "FedGM_MNIST_iid_5_Client__50_epoch"
    os.makedirs(out_dir, exist_ok=True)

    # For demonstration, let's do Dirichlet Non-IID
    client_datasets = dirichlet_noniid_partition(
        train_dataset,
        num_clients=num_clients,
        alpha1=8,
        alpha2=0.1,
        seed=42,
        output_path=os.path.join(out_dir, "client_data_dist.png")
    )
    # iid
    # client_datasets = iid_partition(
    #     train_dataset,
    #     num_clients=num_clients,
    #     output_path=os.path.join(out_dir, "client_data_dist.png")
    # )

    # 3) Federated hyperparams
    global_rounds = 50
    local_epochs  = 2
    batch_size    = 32
    init_lr       = 1.0

    # 4) Initialize global model
    global_model = SmallNet()

    # Logging
    global_accuracies = []                                # Server accuracy each round
    client_accuracies = [[] for _ in range(num_clients)]  # Each client's accuracy after local updates

    # Per-client local epoch logs (loss & LR)
    client_epoch_losses = [[] for _ in range(num_clients)]
    client_epoch_lrs    = [[] for _ in range(num_clients)]

    # We want "learning rate per client per global epoch":
    client_lr_final_per_round = [[] for _ in range(num_clients)]
    client_grad_norms_per_round = [[] for _ in range(num_clients)]

    # (1) Open a TXT file for storing results table
    result_path = os.path.join(out_dir, "result.txt")
    with open(result_path, "w") as f:
        # Write header
        f.write(f"{'GlobalRnd':<12}{'ClientID':<10}{'LocEpoch':<12}{'Accuracy':<12}{'LR':<12}{'Loss':<12}{'GradNorm':<12}\n")
    
        # 5) Federated Rounds
        for rnd in range(global_rounds):
            print(f"\n--- Global Round {rnd+1}/{global_rounds} ---")
            local_models = []

            for cid in range(num_clients):
                print(f"[Client {cid}] local training ...")
                # Copy global model
                client_model = copy.deepcopy(global_model)

                # Local training w/ SASS
                updated_model, epoch_losses, epoch_lrs, avg_grad_norm = local_train_sass(
                    client_model,
                    client_datasets[cid],
                    device,
                    local_epochs=local_epochs,
                    batch_size=batch_size,
                    init_lr=init_lr
                )
                local_models.append(updated_model)

                # Evaluate updated model on test set
                client_acc = evaluate(updated_model, test_dataset, device=device)
                client_accuracies[cid].append(client_acc)
                print(f"  Client {cid} accuracy after local update: {client_acc:.2f}%")

                # Accumulate epoch logs
                client_epoch_losses[cid].extend(epoch_losses)
                client_epoch_lrs[cid].extend(epoch_lrs)

                # final LR for this client in this round = last entry in epoch_lrs
                final_lr_this_round = epoch_lrs[-1]
                client_lr_final_per_round[cid].append(final_lr_this_round)
                client_grad_norms_per_round[cid].append(avg_grad_norm)

                # final training loss (last local epoch's average)
                final_loss_this_round = epoch_losses[-1]

                # Write one row per client (for final local epoch)
                f.write(
                    f"{rnd+1:<12}{cid:<10}{local_epochs:<12}{client_acc:<12.4f}"
                    f"{final_lr_this_round:<12.6f}{final_loss_this_round:<12.6f}{avg_grad_norm:<12.6f}\n"
                )
            
            # -------------------------
            # Geometric Median Aggregate
            # -------------------------
            gm_vector = weiszfeld_geometric_median(local_models)
            set_model_from_vector(global_model, gm_vector)

            # Evaluate global model
            global_acc = evaluate(global_model, test_dataset, device=device)
            global_accuracies.append(global_acc)
            print(f"Global accuracy after round {rnd+1}: {global_acc:.2f}%")
    
    # 6) PLOTS

    # (A) Accuracy (client & global) vs. Global Round
    rounds_range = range(1, global_rounds + 1)
    plt.figure()
    for cid in range(num_clients):
        plt.plot(rounds_range, client_accuracies[cid], linestyle="-", linewidth=2, label=f"Client {cid} Acc")
    plt.plot(rounds_range, global_accuracies, marker='x', linewidth=2, linestyle="--",
             color="red", label="Global Acc")
    plt.title("Client & Global Accuracy vs. Global Round (Geometric Median)")
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_clients_and_global.png"))
    plt.close()

    # (B) Learning Rate per Client per Global Epoch
    plt.figure()
    for cid in range(num_clients):
        plt.plot(rounds_range, client_lr_final_per_round[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client per Global Epoch (Geometric Median)")
    plt.xlabel("Global Round")
    plt.ylabel("Final LR in Local Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_epoch.png"))
    plt.close()

    # (B2) Log-Learning Rate (log10) per Client per Global Epoch
    plt.figure()
    for cid in range(num_clients):
        lr_log = [np.log10(lr) for lr in client_lr_final_per_round[cid]]
        plt.plot(rounds_range, lr_log, linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Log10(Learning Rate) per Client per Global Epoch (Geometric Median)")
    plt.xlabel("Global Round")
    plt.ylabel("log10(Final LR)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_epoch_log.png"))
    plt.close()

    # (C) Per-client Learning Rate per Local Epoch
    plt.figure()
    for cid in range(num_clients):
        plt.plot(range(1, len(client_epoch_lrs[cid]) + 1),
                 client_epoch_lrs[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client (Local Epoch Index)")
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("LR (Step Size)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "learning_rate_per_client_local_epochs.png"))
    plt.close()

    # (D) Per-client Training Loss per Local Epoch
    plt.figure()
    for cid in range(num_clients):
        plt.plot(range(1, len(client_epoch_losses[cid]) + 1),
                 client_epoch_losses[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Training Loss per Client (Local Epoch Index)")
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_loss_per_client.png"))
    plt.close()

    # (E) Gradient Norm per Client per Global Epoch
    plt.figure()
    for cid in range(num_clients):
        plt.plot(rounds_range, client_grad_norms_per_round[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Gradient Norm per Client per Global Epoch (Geometric Median + SASS)")
    plt.xlabel("Global Round")
    plt.ylabel("Avg Gradient Norm (Local Training)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "grad_norm_per_client_per_global_epoch.png"))
    plt.close()

    print(f"\nPlots saved in folder: '{out_dir}'")
    print("Text table saved to:   '{}'".format(os.path.join(out_dir, 'result.txt')))
    print("You have:")
    print("  - client_data_dist.png")
    print("  - accuracy_clients_and_global.png")
    print("  - lr_per_client_per_global_epoch.png")
    print("  - lr_per_client_per_global_epoch_log.png   (log-scale LR)")
    print("  - learning_rate_per_client_local_epochs.png")
    print("  - training_loss_per_client.png")
    print("  - grad_norm_per_client_per_global_epoch.png")
    print("  - result.txt (table of round-wise results)")

if __name__ == "__main__":
    main()
