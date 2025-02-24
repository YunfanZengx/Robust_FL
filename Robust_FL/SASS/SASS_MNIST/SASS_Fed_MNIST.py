
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from sass import Sass

data_dir = './mnist_data'
os.makedirs(data_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

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
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class FedSASSClient:
    def __init__(self, client_id, train_data, batch_size=32, local_epochs=2, alpha_0=1.0, gamma=0.7, theta=0.2, eps_f=1e-2):
        self.client_id = client_id
        self.train_data = train_data
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.alpha_0 = alpha_0
        self.gamma = gamma
        self.theta = theta
        self.eps_f = eps_f
        self.loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.model = SmallNet()
        self.learning_rates = []  # Track learning rate per epoch
        self.grad_norms = []  # Track gradient norm per epoch
    
    def train(self, global_model_state):
        self.model.load_state_dict(global_model_state)
        optimizer = Sass(self.model.parameters(), n_batches_per_epoch=len(self.loader), init_step_size=self.alpha_0, theta=self.theta, gamma_decr=self.gamma, eps_f=self.eps_f)
        for epoch in range(self.local_epochs):
            epoch_lr = []
            epoch_grad_norm = []
            for data, target in self.loader:
                def closure():
                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = F.nll_loss(output, target)
                    return loss
                optimizer.step(closure)
                epoch_lr.append(optimizer.state['step_size'])
                epoch_grad_norm.append(optimizer.state['grad_norm'].item())
            self.learning_rates.append(np.mean(epoch_lr))
            self.grad_norms.append(np.mean(epoch_grad_norm))
        return self.model.state_dict()

class FedSASSServer:
    def __init__(self, n_clients, imbalance_factor=0.5):
        self.n_clients = n_clients
        self.global_model = SmallNet()
        self.global_model_state = self.global_model.state_dict()
        self.clients = self.create_clients(imbalance_factor)
        self.client_learning_rates = [[] for _ in range(n_clients)]
        self.client_grad_norms = [[] for _ in range(n_clients)]

    def create_clients(self, imbalance_factor):
        n_samples = len(train_dataset) // self.n_clients
        min_samples = int((1 - imbalance_factor) * n_samples)
        max_samples = int((1 + imbalance_factor) * n_samples)
        client_datasets = []
        for _ in range(self.n_clients):
            n_client_samples = np.random.randint(min_samples, max_samples + 1)
            client_subset, _ = random_split(train_dataset, [n_client_samples, len(train_dataset) - n_client_samples])
            client_datasets.append(client_subset)
        return [FedSASSClient(i, data) for i, data in enumerate(client_datasets)]

    def aggregate_models(self, client_models, tol=1e-4, max_iter=100):
        # Convert state_dicts to flattened tensors
        flattened_models = [torch.cat([v.flatten() for v in model.values()]) for model in client_models]
        
        # Initialize the median as the mean of the flattened models
        median = torch.mean(torch.stack(flattened_models), dim=0)
        
        for iteration in range(max_iter):
            distances = torch.tensor([torch.norm(flat_model - median) for flat_model in flattened_models])
            distances = torch.clamp(distances, min=1e-10)
            weights = 1.0 / distances
            new_median = torch.sum(torch.stack([w * flat_model for w, flat_model in zip(weights, flattened_models)]), dim=0) / torch.sum(weights)
            
            if torch.norm(new_median - median) < tol:
                break
            
            median = new_median
        
        # Convert the flattened median back to a state_dict
        new_state_dict = {}
        idx = 0
        for k, v in client_models[0].items():
            num_elements = v.numel()
            new_state_dict[k] = new_median[idx:idx + num_elements].view_as(v)
            idx += num_elements
        
        return new_state_dict


    def train(self, num_epochs):
        accuracy_list = []
        for epoch in range(num_epochs):
            client_models = []
            for i, client in enumerate(self.clients):
                client_model = client.train(self.global_model_state)
                client_models.append(client_model)
                self.client_learning_rates[i].append(np.mean(client.learning_rates))
                self.client_grad_norms[i].append(np.mean(client.grad_norms))
            self.global_model_state = self.aggregate_models(client_models)
            self.global_model.load_state_dict(self.global_model_state)
            accuracy = self.evaluate()
            accuracy_list.append(accuracy)
            print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")
        return accuracy_list

    def evaluate(self):
        self.global_model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(test_loader.dataset)

n_clients = 5
server = FedSASSServer(n_clients=n_clients, imbalance_factor=0.5)
num_epochs = 10
accuracy_list = server.train(num_epochs)

output_dir = "./output_FedSASS_MNIST_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), accuracy_list, marker='o', linestyle='-', color='g')
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy over Epochs for Federated SASS with MNIST")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "test_accuracy.png"))
plt.close()

plt.figure(figsize=(8, 6))
for i in range(n_clients):
    plt.plot(range(1, num_epochs + 1), server.client_learning_rates[i], marker='o', linestyle='-', label=f"Client {i+1}")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate per Client per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "learning_rate_per_client.png"))
plt.close()

plt.figure(figsize=(8, 6))
for i in range(n_clients):
    plt.plot(range(1, num_epochs + 1), server.client_grad_norms[i], marker='o', linestyle='-', label=f"Client {i+1}")
plt.xlabel("Epoch")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm per Client per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, "grad_norm_per_client.png"))
plt.close()
