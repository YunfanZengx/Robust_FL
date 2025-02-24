import os
import numpy as np
import torch
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def simple_distribution_plot(num_clients, num_classes, distribution, filename):
    """
    Draw a horizontal stacked bar chart showing the distribution across clients.
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
    ax.barh(range(num_clients), dist_array[0], color=colors[0])
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
    """
    Create two plots (vertical bar and horizontal stacked) showing the data distribution.
    """
    class_counts = [torch.zeros(num_classes, dtype=torch.int32) for _ in range(num_clients)]
    for i, dataset in enumerate(client_datasets):
        labels = [label for _, label in dataset]
        unique, counts = torch.tensor(labels).unique(return_counts=True)
        for cls, count in zip(unique, counts):
            class_counts[i][cls] = count
    plt.figure(figsize=(10, 6))
    for i in range(num_clients):
        plt.bar(np.arange(num_classes) + i * 0.1, class_counts[i],
                width=0.1, label=f'Client {i+1}')
    plt.xlabel("Class Label")
    plt.ylabel("Number of Samples")
    plt.title("Data Distribution Across Clients (Plot A)")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    distribution_list = [cc.tolist() for cc in class_counts]
    output_path_b = output_path.replace(".png", "_B.png")
    simple_distribution_plot(num_clients, num_classes, distribution_list, output_path_b)

def iid_partition(train_dataset, num_clients, output_path=None):
    num_samples = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_datasets = []
    for i in range(num_clients):
        start = i * num_samples
        end = (i + 1) * num_samples
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
        num_samples_per_client = np.maximum(1, (client_class_proportions[c] * len(cinds)).astype(int))
        cum_samples = np.cumsum(num_samples_per_client)
        start_idx = 0
        for i in range(num_clients):
            end_idx = min(cum_samples[i], len(cinds))
            if start_idx < end_idx:
                client_indices[i].extend(cinds[start_idx:end_idx])
            start_idx = end_idx
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
    Each client gets data from exactly one distinct class.
    """
    np.random.seed(seed)
    all_labels = [label for _, label in train_dataset]
    unique_labels = list(set(all_labels))
    num_classes = len(unique_labels)
    if num_clients > num_classes:
        raise ValueError(f"For 'extreme' partition, number of clients ({num_clients}) cannot exceed number of classes ({num_classes}).")
    np.random.shuffle(unique_labels)
    chosen_labels = unique_labels[:num_clients]
    class_indices = {c: [] for c in chosen_labels}
    for idx, (_, label) in enumerate(train_dataset):
        if label in chosen_labels:
            class_indices[label].append(idx)
    client_datasets = []
    for i in range(num_clients):
        label_i = chosen_labels[i]
        subset_i = Subset(train_dataset, class_indices[label_i])
        client_datasets.append(subset_i)
    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    return client_datasets

def case1_partition(train_dataset, num_clients, seed=42, output_path=None):
    """
    Partition strategy: Client 0 gets 600 samples from class 0 and the others get evenly split remaining data.
    """
    np.random.seed(seed)
    class0_indices = [idx for idx, (_, label) in enumerate(train_dataset) if label == 0]
    np.random.shuffle(class0_indices)
    client0_indices = class0_indices[:600]
    remaining_indices = class0_indices[600:]
    other_indices = [idx for idx, (_, label) in enumerate(train_dataset) if label != 0]
    combined_remaining = remaining_indices + other_indices
    np.random.shuffle(combined_remaining)
    num_remaining_clients = num_clients - 1
    if num_remaining_clients > 0:
        splits = np.array_split(combined_remaining, num_remaining_clients)
    else:
        splits = [combined_remaining]
    client_datasets = [Subset(train_dataset, client0_indices)]
    for split in splits:
        client_datasets.append(Subset(train_dataset, split))
    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    return client_datasets

def case2_partition(train_dataset, num_clients, seed=42, output_path=None):
    """
    Partition strategy (case2): 
      - IID partition among clients.
      - Within client 0, 30% of class 0 and class 1 samples are mislabeled.
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    # IID partition
    num_samples = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_datasets = []
    for i in range(num_clients):
        start = i * num_samples
        end = (i + 1) * num_samples
        subset_indices = indices[start:end]
        client_datasets.append(Subset(train_dataset, subset_indices))
    
    # Mislabel 30% of class 0 and 1 samples in client 0
    client0_dataset = client_datasets[0]
    client0_indices = list(client0_dataset.indices)
    class0_indices = [idx for idx in client0_indices if train_dataset[idx][1] == 0]
    class1_indices = [idx for idx in client0_indices if train_dataset[idx][1] == 1]
    
    num_mislabel_class0 = int(0.3 * len(class0_indices))
    num_mislabel_class1 = int(0.3 * len(class1_indices))
    
    mislabel_class0_indices = np.random.choice(class0_indices, num_mislabel_class0, replace=False)
    mislabel_class1_indices = np.random.choice(class1_indices, num_mislabel_class1, replace=False)
    
    for idx in mislabel_class0_indices:
        wrong_label = random.choice(list(range(1, 10)))
        train_dataset.targets[idx] = wrong_label
    
    for idx in mislabel_class1_indices:
        wrong_label = random.choice([0] + list(range(2, 10)))
        train_dataset.targets[idx] = wrong_label
    
    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    
    return client_datasets

def case3_partition(train_dataset, num_clients, seed=42, output_path=None):
    """
    Partition strategy (case3): 
      - First, perform an IID partition among all clients (like case2).
      - Then, for:
          * Client 0: mislabel 30% of samples that belong to classes 0 and 1.
          * Client 1: mislabel 40% of samples that belong to classes 2 and 3.
          * Client 2: mislabel 30% of samples that belong to classes 4, 5, 6, 7, and 8.
      - All other clients (if any) remain IID.
    """
    import random
    np.random.seed(seed)
    random.seed(seed)
    # First, get an IID partition among all clients.
    client_datasets = iid_partition(train_dataset, num_clients, output_path=None)
    
    # --- Client 0 modifications: classes 0 and 1, 30% mislabeling ---
    if num_clients >= 1:
        client0_dataset = client_datasets[0]
        client0_indices = list(client0_dataset.indices)
        class0_indices = [idx for idx in client0_indices if train_dataset[idx][1] == 0]
        class1_indices = [idx for idx in client0_indices if train_dataset[idx][1] == 1]
        num_mislabel_class0 = int(0.3 * len(class0_indices))
        num_mislabel_class1 = int(0.3 * len(class1_indices))
        if num_mislabel_class0 > 0:
            mislabel_class0_indices = np.random.choice(class0_indices, num_mislabel_class0, replace=False)
            for idx in mislabel_class0_indices:
                wrong_label = random.choice([l for l in range(10) if l != 0])
                train_dataset.targets[idx] = wrong_label
        if num_mislabel_class1 > 0:
            mislabel_class1_indices = np.random.choice(class1_indices, num_mislabel_class1, replace=False)
            for idx in mislabel_class1_indices:
                wrong_label = random.choice([l for l in range(10) if l != 1])
                train_dataset.targets[idx] = wrong_label

    # --- Client 1 modifications: classes 2 and 3, 40% mislabeling ---
    if num_clients >= 2:
        client1_dataset = client_datasets[1]
        client1_indices = list(client1_dataset.indices)
        class2_indices = [idx for idx in client1_indices if train_dataset[idx][1] == 2]
        class3_indices = [idx for idx in client1_indices if train_dataset[idx][1] == 3]
        num_mislabel_class2 = int(0.4 * len(class2_indices))
        num_mislabel_class3 = int(0.4 * len(class3_indices))
        if num_mislabel_class2 > 0:
            mislabel_class2_indices = np.random.choice(class2_indices, num_mislabel_class2, replace=False)
            for idx in mislabel_class2_indices:
                wrong_label = random.choice([l for l in range(10) if l != 2])
                train_dataset.targets[idx] = wrong_label
        if num_mislabel_class3 > 0:
            mislabel_class3_indices = np.random.choice(class3_indices, num_mislabel_class3, replace=False)
            for idx in mislabel_class3_indices:
                wrong_label = random.choice([l for l in range(10) if l != 3])
                train_dataset.targets[idx] = wrong_label

    # --- Client 2 modifications: classes 4,5,6,7,8, 30% mislabeling ---
    if num_clients >= 3:
        client2_dataset = client_datasets[2]
        client2_indices = list(client2_dataset.indices)
        target_classes = [4, 5, 6, 7, 8]
        class_indices = [idx for idx in client2_indices if train_dataset[idx][1] in target_classes]
        num_mislabel = int(0.3 * len(class_indices))
        if num_mislabel > 0 and len(class_indices) > 0:
            mislabel_indices = np.random.choice(class_indices, num_mislabel, replace=False)
            for idx in mislabel_indices:
                current_label = train_dataset[idx][1]
                wrong_label = random.choice([l for l in range(10) if l != current_label])
                train_dataset.targets[idx] = wrong_label

    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    return client_datasets
