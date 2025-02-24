
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, Subset

def plot_distribution(client_datasets, num_clients, num_classes, output_path):
    """ Visualize the distribution of classes across clients """
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
    """ IID partition: equally partition the dataset across clients """
    num_samples = len(train_dataset) // num_clients
    indices = np.random.permutation(len(train_dataset))
    client_datasets = [Subset(train_dataset, indices[i * num_samples: (i + 1) * num_samples]) for i in range(num_clients)]

    if output_path:
        plot_distribution(client_datasets, num_clients, 10, output_path)
    
    return client_datasets

def class_based_noniid_partition(train_dataset, num_clients, seed=42, output_path=None):
    """ Class-based Non-IID partition: each client gets a random subset of classes """
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {i: [] for i in labels}
    
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)
    
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    
    client_datasets = [[] for _ in range(num_clients)]
    for cls in labels:
        num_samples = len(class_indices[cls])
        split_indices = np.array_split(class_indices[cls], num_clients)
        for i in range(num_clients):
            client_datasets[i].extend(split_indices[i])
    
    client_datasets = [Subset(train_dataset, client_datasets[i]) for i in range(num_clients)]

    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    
    return client_datasets

def dirichlet_noniid_partition(train_dataset, num_clients, alpha1=8, alpha2=0.5, seed=42, output_path=None):
    """ Dirichlet-based Non-IID partition: ensure at least one sample per client and handle edge cases """
    np.random.seed(seed)
    labels = set([label for _, label in train_dataset])
    num_classes = len(labels)
    class_indices = {i: [] for i in labels}
    
    # Split dataset by class
    for idx, (_, label) in enumerate(train_dataset):
        class_indices[label].append(idx)

    # Shuffle indices within each class
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])

    # Generate class proportions for each client using Dirichlet(alpha2)
    client_class_proportions = np.random.dirichlet([alpha2] * num_clients, num_classes)
    
    # Allocate samples ensuring no empty datasets
    client_datasets = [[] for _ in range(num_clients)]
    for cls in labels:
        class_samples = class_indices[cls]
        num_samples_per_client = np.maximum(1, (client_class_proportions[cls] * len(class_samples)).astype(int))
        cumulative_samples = np.cumsum(num_samples_per_client)
        
        start_idx = 0
        for i in range(num_clients):
            end_idx = min(cumulative_samples[i], len(class_samples))
            if start_idx < end_idx:  # Ensure non-zero samples are allocated
                client_datasets[i].extend(class_samples[start_idx:end_idx])
            start_idx = end_idx

    # Ensure no client dataset is empty by checking length and redistributing if necessary
    for i, dataset in enumerate(client_datasets):
        if len(dataset) == 0:
            # Find the client with the most samples and transfer one sample
            max_client_idx = np.argmax([len(d) for d in client_datasets])
            client_datasets[i].append(client_datasets[max_client_idx].pop())

    client_datasets = [Subset(train_dataset, client_datasets[i]) for i in range(num_clients)]

    if output_path:
        plot_distribution(client_datasets, num_clients, num_classes, output_path)
    
    return client_datasets
