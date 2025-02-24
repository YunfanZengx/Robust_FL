import os
import copy
import argparse
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from partition import (
    iid_partition,
    class_based_noniid_partition,
    dirichlet_noniid_partition,
    extreme_noniid_partition,
    case1_partition,
    case2_partition,
    case3_partition,
)
from model import SmallNet, CIFARNet
from client import local_train_sass, local_train_sgd, evaluate_loss, Sass
from aggregation import fedavg_aggregate, fed_gm_aggregate



# python /home/local/ASURITE/yzeng88/Robust_FL/main/main.py --dataset cifar10 --num_clients 3 --global_rounds 5 --local_epochs 1 --dist case2 --agg_method gm --local_opt sass


def evaluate(model, test_dataset, device="cpu", batch_size=1000):
    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clients", type=int, default=2, help="number of clients")
    parser.add_argument("--global_rounds", type=int, default=5, help="number of global rounds")
    parser.add_argument("--local_epochs", type=int, default=1, help="local epochs per round")
    parser.add_argument("--batch_size", type=int, default=32, help="local batch size")
    parser.add_argument("--dist", type=str, default="iid",
                        choices=["iid", "class", "dirichlet", "extreme", "case1", "case2", "case3"],
                        help="data distribution")
    parser.add_argument("--alpha2", type=float, default=0.5,
                        help="Dirichlet alpha2 parameter (if dist=dirichlet)")
    parser.add_argument("--agg_method", type=str, default="gm", choices=["avg", "gm"],
                        help="Aggregation method: avg for FedAvg, gm for FedGM")
    parser.add_argument("--local_opt", type=str, default="sass", choices=["sass", "sgd"],
                        help="Local optimizer: sass for SASS, sgd for SGD")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for SGD (ignored if local_opt is sass)")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to use: 'mnist' or 'cifar10'")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load the dataset
    if args.dataset == "mnist":
        data_dir = "./mnist_data"
        os.makedirs(data_dir, exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        global_model = SmallNet()
    elif args.dataset == "cifar10":
        data_dir = "./cifar_data"
        os.makedirs(data_dir, exist_ok=True)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        global_model = CIFARNet()
    else:
        raise ValueError("Unknown dataset specified!")

    # 2) Partition data
    if args.dataset == "mnist":
        main_folder = "A_experiment_output_mnist"
    elif args.dataset == "cifar10":
        main_folder = "A_experiment_output_cifar10"
    else:
        main_folder = "A_experiment_output"
    os.makedirs(main_folder, exist_ok=True)
    out_dir_name = f"{'fedavg' if args.agg_method=='avg' else 'fedgm'}_{args.local_opt}_{args.dist}_{args.num_clients}clients_{args.global_rounds}rounds"
    out_dir = os.path.join(main_folder, out_dir_name)
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
    elif args.dist == "case1":
        client_datasets = case1_partition(train_dataset, args.num_clients,
                                          seed=seed,
                                          output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "case2":
        client_datasets = case2_partition(train_dataset, args.num_clients,
                                          seed=seed,
                                          output_path=os.path.join(out_dir, "data_dist.png"))
    elif args.dist == "case3":
        client_datasets = case3_partition(train_dataset, args.num_clients,
                                          seed=seed,
                                          output_path=os.path.join(out_dir, "data_dist.png"))
    else:
        raise ValueError("Unknown data distribution type!")

    # 3) Create a local optimizer per client
    client_optimizers = []
    for cid in range(args.num_clients):
        if args.local_opt == "sass":
            opt = Sass(global_model.parameters(),
                       init_step_size=1.0,
                       theta=0.2,
                       gamma_decr=0.7,
                       gamma_incr=1.25,
                       alpha_max=10.0,
                       eps_f=0.0)
        elif args.local_opt == "sgd":
            opt = torch.optim.SGD(global_model.parameters(), lr=args.lr)
        else:
            raise ValueError("Unknown local optimizer!")
        client_optimizers.append(opt)

    # 4) Logging structures
    global_accuracies = []
    client_accuracies = [[] for _ in range(args.num_clients)]
    client_epoch_losses = [[] for _ in range(args.num_clients)]
    client_epoch_lrs    = [[] for _ in range(args.num_clients)]
    client_lr_final_per_round = [[] for _ in range(args.num_clients)]
    client_grad_norms_per_round = [[] for _ in range(args.num_clients)]

    nn_passes_counters = [0] * args.num_clients
    train_loss_vs_passes = [[] for _ in range(args.num_clients)]
    test_loss_vs_passes  = [[] for _ in range(args.num_clients)]

    result_path = os.path.join(out_dir, "result.txt")
    server_acc_path = os.path.join(out_dir, "server_accuracy.txt")

    with open(result_path, "w") as f, open(server_acc_path, "w") as f_server:
        header = f"{'GlobalRnd':<12}{'ClientID':<10}{'LocEpoch':<12}{'Accuracy':<12}"\
                 f"{'LR':<12}{'Loss':<12}{'GradNorm':<12}\n"
        f.write(header)
        f_server.write("GlobalRound\tGlobalAccuracy\n")
        for rnd in range(args.global_rounds):
            print(f"\n--- Global Round {rnd+1}/{args.global_rounds} ---")
            local_models = []
            for cid in range(args.num_clients):
                print(f"[Client {cid}] local training ...")
                # Deepcopy the global model for client update
                client_model = copy.deepcopy(global_model)
                if args.local_opt == "sass":
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
                elif args.local_opt == "sgd":
                    client_optimizers[cid].param_groups[0]["params"] = list(client_model.parameters())
                    updated_model, epoch_losses, epoch_lrs, _ = local_train_sgd(
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
                    avg_grad_norm = 0.0
                local_models.append(updated_model)
                cacc = evaluate(updated_model, test_dataset, device=device)
                client_accuracies[cid].append(cacc)
                print(f"  Client {cid} accuracy after local update: {cacc:.2f}%")
                client_epoch_losses[cid].extend(epoch_losses)
                client_epoch_lrs[cid].extend(epoch_lrs)
                final_lr_this_round = epoch_lrs[-1]
                final_loss_this_round = epoch_losses[-1]
                client_lr_final_per_round[cid].append(final_lr_this_round)
                client_grad_norms_per_round[cid].append(avg_grad_norm)
                f.write(f"{rnd+1:<12}{cid:<10}{args.local_epochs:<12}{cacc:<12.4f}"
                        f"{final_lr_this_round:<12.6f}{final_loss_this_round:<12.6f}{avg_grad_norm:<12.6f}\n")
            # 5) Global aggregation
            if args.num_clients > 1:
                if args.agg_method == "avg":
                    global_model = fedavg_aggregate(global_model, local_models)
                elif args.agg_method == "gm":
                    global_model = fed_gm_aggregate(global_model, local_models, tol=1e-4, max_iter=100)
                else:
                    raise ValueError("Unknown aggregation method!")
            else:
                global_model.load_state_dict(local_models[0].state_dict())
            global_acc = evaluate(global_model, test_dataset, device=device)
            global_accuracies.append(global_acc)
            print(f"Global accuracy after round {rnd+1}: {global_acc:.2f}%")
            f_server.write(f"{rnd+1}\t{global_acc:.4f}\n")

    # 6) Plotting results
    rounds_range = range(1, args.global_rounds + 1)
    title_suffix = f" ({'FedAvg' if args.agg_method=='avg' else 'FedGM'} + {args.local_opt.upper()})"
    
    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_accuracies[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.plot(rounds_range, global_accuracies, marker='x', linestyle="--", linewidth=2, color="red", label="Global Acc")
    plt.title("Client & Global Accuracy vs. Global Round" + title_suffix)
    plt.xlabel("Global Round")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "accuracy_clients_and_global.png"))
    plt.close()

    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_lr_final_per_round[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client per Global Round" + title_suffix)
    plt.xlabel("Global Round")
    plt.ylabel("Final LR in Local Training")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_round.png"))
    plt.close()

    plt.figure()
    for cid in range(args.num_clients):
        lr_log = [np.log10(lr) for lr in client_lr_final_per_round[cid]]
        plt.plot(rounds_range, lr_log, linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Log10(LR) per Client per Global Round" + title_suffix)
    plt.xlabel("Global Round")
    plt.ylabel("log10(Final LR)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "lr_per_client_per_global_round_log.png"))
    plt.close()

    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(range(1, len(client_epoch_lrs[cid]) + 1), client_epoch_lrs[cid],
                 linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Learning Rate per Client (Local Epoch Index)" + title_suffix)
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("LR (Step Size)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "learning_rate_per_client_local_epochs.png"))
    plt.close()

    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(range(1, len(client_epoch_losses[cid]) + 1), client_epoch_losses[cid],
                 linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Training Loss per Client (Local Epoch Index)" + title_suffix)
    plt.xlabel("Local Epoch Index (Cumulative)")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "training_loss_per_client.png"))
    plt.close()

    plt.figure()
    for cid in range(args.num_clients):
        plt.plot(rounds_range, client_grad_norms_per_round[cid], linestyle="-", linewidth=2, label=f"Client {cid}")
    plt.title("Gradient Norm per Client per Global Round" + title_suffix)
    plt.xlabel("Global Round")
    plt.ylabel("Avg Gradient Norm (Local Training)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "grad_norm_per_client_per_global_round.png"))
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=args.num_clients, figsize=(6 * args.num_clients, 5))
    if args.num_clients == 1:
        axes = [axes]
    for cid in range(args.num_clients):
        if len(train_loss_vs_passes[cid]) > 0:
            passes, losses = zip(*train_loss_vs_passes[cid])
            axes[cid].plot(passes, losses, color='blue', linewidth=1)
            axes[cid].set_title(f"Client {cid} - Train Loss vs NN Passes")
            axes[cid].set_xlabel("NN Passes")
            axes[cid].set_ylabel("Train Loss")
            axes[cid].set_yscale("log")
            axes[cid].yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
            axes[cid].yaxis.set_major_formatter(ticker.LogFormatterMathtext())
            axes[cid].grid(True)
        else:
            axes[cid].set_title(f"Client {cid} - No Train Data")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_loss_vs_nn_passes_per_client.png"))
    plt.close()

    fig, axes = plt.subplots(nrows=1, ncols=args.num_clients, figsize=(6 * args.num_clients, 5))
    if args.num_clients == 1:
        axes = [axes]
    for cid in range(args.num_clients):
        if len(test_loss_vs_passes[cid]) > 0:
            passes, losses = zip(*test_loss_vs_passes[cid])
            axes[cid].plot(passes, losses, color='red', marker='o', linewidth=1)
            axes[cid].set_title(f"Client {cid} - Test Loss vs NN Passes")
            axes[cid].set_xlabel("NN Passes")
            axes[cid].set_ylabel("Test Loss")
            axes[cid].set_yscale("log")
            axes[cid].yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
            axes[cid].yaxis.set_major_formatter(ticker.LogFormatterMathtext())
            axes[cid].grid(True)
        else:
            axes[cid].set_title(f"Client {cid} - No Test Data")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "test_loss_vs_nn_passes_per_client.png"))
    plt.close()

    loss_file_path = os.path.join(out_dir, "loss.txt")
    with open(loss_file_path, "w") as fout:
        fout.write("ClientID\tNNPass_Train\tTrainLoss\tNNPass_Test\tTestLoss\n")
        for cid in range(args.num_clients):
            T = len(train_loss_vs_passes[cid])
            U = len(test_loss_vs_passes[cid])
            max_len = max(T, U)
            for i in range(max_len):
                if i < T:
                    pt, lt = train_loss_vs_passes[cid][i]
                else:
                    pt, lt = (-1, float('nan'))
                if i < U:
                    pu, lu = test_loss_vs_passes[cid][i]
                else:
                    pu, lu = (-1, float('nan'))
                fout.write(f"{cid}\t{pt}\t{lt:.6f}\t{pu}\t{lu:.6f}\n")
    print(f"Loss vs NN passes data saved to: '{loss_file_path}'")
    print(f"\nPlots saved in folder: '{out_dir}'")
    print(f"Client & server metrics saved to: '{result_path}' and '{server_acc_path}'")

if __name__ == "__main__":
    main()
