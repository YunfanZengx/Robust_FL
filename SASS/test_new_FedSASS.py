import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from sass import Sass  

############################
# 1. Federation Parameters #
############################
num_clients = 3  # Example: set to 1 for single-client or >1 for multi-client
num_global_rounds = 50  # Number of total global rounds
local_epochs = 1       # How many epochs each client trains per round

############################
# 2. Data Preparation      #
############################
data_dir = './mnist_data'
os.makedirs(data_dir, exist_ok=True)
save_path = "new_Fed_sass_result_3_clients"  # Adjust if needed
os.makedirs(save_path, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the full MNIST training set
full_train_dataset = torchvision.datasets.MNIST(
    data_dir, train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    data_dir, train=False, download=True, transform=transform
)

# If num_clients > 1, we'll split the dataset among them; if ==1, single dataset.
dataset_size = len(full_train_dataset)
per_client_size = dataset_size // num_clients
client_datasets = []
start = 0
for c in range(num_clients):
    # Last client gets remainder
    end = start + per_client_size if c < num_clients - 1 else dataset_size
    subset_ds = torch.utils.data.Subset(full_train_dataset, range(start, end))
    client_datasets.append(subset_ds)
    start = end

test_loader = DataLoader(test_dataset, batch_size=500, shuffle=False)

##########################
# 3. Model Definition    #
##########################
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

##########################
# 4. Helper Functions    #
##########################
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
    A simple FedAvg aggregator: 
    Takes a list of model replicas and returns an *in-place* averaged state_dict.
    """
    base_model = list_of_models[0]
    base_params = base_model.state_dict()
    
    for key in base_params.keys():
        for i in range(1, len(list_of_models)):
            base_params[key] += list_of_models[i].state_dict()[key]
        base_params[key] = base_params[key] / len(list_of_models)
    
    return base_params

##############################
# 5. Global Federated Loop   #
##############################
# Initialize the *global* model
global_model = SmallNet()
global_model_weights = global_model.state_dict()

# Tracking stats
train_losses_over_rounds = []
test_losses_over_rounds = []
accuracy_over_rounds = []

# --------------------------------------------------
# Instead of a single lr_per_round, track per-client
# --------------------------------------------------
lr_per_client_per_round = [[] for _ in range(num_clients)]
# This will let us plot each client's LR curve (vs. global rounds)

for round_idx in range(num_global_rounds):
    print(f"\n--- Global Round {round_idx+1}/{num_global_rounds} ---")
    
    local_models = []
    local_avg_losses = []
    
    # ----- Each client trains locally -----
    for client_idx in range(num_clients):
        # 1) Create a local model and copy the global weights
        local_model = SmallNet()
        local_model.load_state_dict(copy.deepcopy(global_model_weights))
        
        # 2) Create client dataloader
        local_train_dataset = client_datasets[client_idx]
        local_train_loader = DataLoader(local_train_dataset, batch_size=32, shuffle=False)
        
        # 3) SASS optimizer
        n_batches_per_epoch = len(local_train_loader)
        local_optimizer = Sass(local_model.parameters(), n_batches_per_epoch=n_batches_per_epoch)
        
        # 4) Train for `local_epochs`
        for ep in range(local_epochs):
            avg_loss = train_one_epoch(local_model, local_train_loader, local_optimizer)

        local_models.append(local_model)
        local_avg_losses.append(avg_loss)
        
        # 5) Grab the *last step size* used by this client's SASS
        if local_optimizer.state['step_size_vs_nn_passes']:
            last_step_size = local_optimizer.state['step_size_vs_nn_passes'][-1][1]
        else:
            last_step_size = float('nan')

        # Store this in lr_per_client_per_round[client_idx]
        lr_per_client_per_round[client_idx].append(last_step_size)
    
    # ----- Aggregate if more than one client -----
    if num_clients > 1:
        new_weights = average_weights(local_models)
        global_model.load_state_dict(new_weights)
        global_model_weights = copy.deepcopy(new_weights)
    else:
        # If only one client, global = local
        global_model.load_state_dict(local_models[0].state_dict())
        global_model_weights = copy.deepcopy(local_models[0].state_dict())

    # Calculate average local train loss for logging
    mean_train_loss = sum(local_avg_losses) / len(local_avg_losses)
    train_losses_over_rounds.append(mean_train_loss)

    # Evaluate on test set
    test_loss, test_acc = evaluate(global_model, test_loader)
    test_losses_over_rounds.append(test_loss)
    accuracy_over_rounds.append(test_acc)

    print(f"Round {round_idx+1} - Avg local train loss: {mean_train_loss:.4f}")
    print(f"Round {round_idx+1} - Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

# ---------------------------------------------------
# 6. Plot the aggregated training loss across rounds
# ---------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_global_rounds + 1), train_losses_over_rounds, 'o-b')
plt.xlabel("Global Round")
plt.ylabel("Average Local Training Loss")
plt.title("Federated - Avg Local Training Loss vs. Global Round")
plt.grid(True)
plt.savefig(os.path.join(save_path, "federated_train_loss.png"))
plt.close()

# -- Plot test accuracy vs global rounds --
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_global_rounds + 1), accuracy_over_rounds, 'o-g')
plt.xlabel("Global Round")
plt.ylabel("Test Accuracy (%)")
plt.title("Federated - Global Model Test Accuracy vs. Global Round")
plt.grid(True)
plt.savefig(os.path.join(save_path, "federated_test_accuracy.png"))
plt.close()

# ----------------------------------------------------------
# NEW: Plot each client's step size vs. global rounds
# ----------------------------------------------------------
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
