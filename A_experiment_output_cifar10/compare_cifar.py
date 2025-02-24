#!/usr/bin/env python3
"""
compare_all_experiments.py

This script compares CIFAR-10 experiment results from three experiments:
  2) dirichlet non-iid
  3) case2
  4) case3

For each experiment, we produce several pages in the output PDF:

  Page A: Test Loss vs. NN Pass subplots  
    - For experiments 2, 3, and 4 the four algorithms are compared:
         FedGM_SASS (red), FedGM_SGD (blue), FedAvg_SASS (purple), FedAvg_SGD (green)
      in a single row of subplots (one per client), with log-scale on the y-axis.

  Page B: Server Accuracy vs. Global Round  
    - A single plot for each experiment with external accuracy text above the plot.
  
  Extra Page: Training Loss vs. NN Passes  
    - For experiments 2, 3, and 4 (loss.txt contains five columns) an extra page is produced that plots Training Loss vs. NN Passes.

  Extra Page: LR per Client vs. Global Round (log-scale)  
    - Displays the lr_per_client_per_global_round_log.png image from the FedGM_SASS and FedAvg_SASS folders.

All figures are combined into a multi-page PDF:
  /home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/compare_all_experiments.pdf
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg  # for reading images

###############################################################################
# Experiment directories (using A_experiment_output_cifar10)
###############################################################################
# (Note: For CIFAR-10 you ran experiments for distributions: dirichlet, case2, and case3)

# Experiment 2: dirichlet non-iid (4 algorithms)
fedgm_sass_dir_exp2 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sass_dirichlet_5clients_50rounds"
fedgm_sgd_dir_exp2  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sgd_dirichlet_5clients_50rounds"
fedavg_sass_dir_exp2 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sass_dirichlet_5clients_50rounds"
fedavg_sgd_dir_exp2  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sgd_dirichlet_5clients_50rounds"

# Experiment 3: case2 (4 algorithms)
fedgm_sass_dir_exp3 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sass_case2_5clients_50rounds"  # FedGM_SASS
fedgm_sgd_dir_exp3  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sgd_case2_5clients_50rounds"   # FedGM_SGD
fedavg_sass_dir_exp3 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sass_case2_5clients_50rounds"  # FedAvg_SASS
fedavg_sgd_dir_exp3  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sgd_case2_5clients_50rounds"   # FedAvg_SGD

# Experiment 4: case3 (4 algorithms; note: in case3 only clients 0-2 have mislabeling)
fedgm_sass_dir_exp4 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sass_case3_5clients_50rounds"
fedgm_sgd_dir_exp4  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedgm_sgd_case3_5clients_50rounds"
fedavg_sass_dir_exp4 = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sass_case3_5clients_50rounds"
fedavg_sgd_dir_exp4  = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/fedavg_sgd_case3_5clients_50rounds"

###############################################################################
# 1) Reading "loss.txt"
###############################################################################
def read_loss_file(file_path):
    """
    Reads a 'loss.txt' file and returns two dictionaries:
      - data_train: maps client_id -> list of (NNPass_Train, TrainLoss)
      - data_test  : maps client_id -> list of (NNPass_Test, TestLoss)
    
    If the file contains only 3 columns (ClientID, NNPass_Test, TestLoss),
    then data_train will be empty.
    Otherwise (if 5 columns are present) both training and test data are returned.
    """
    data_train = {}
    data_test = {}
    with open(file_path, "r") as f:
        header = f.readline().strip().split()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cid = int(parts[0])
                nn_pass_train = float(parts[1])
                train_loss = float(parts[2])
                nn_pass_test = float(parts[3])
                test_loss = float(parts[4])
                if cid not in data_train:
                    data_train[cid] = []
                    data_test[cid] = []
                data_train[cid].append((nn_pass_train, train_loss))
                data_test[cid].append((nn_pass_test, test_loss))
            elif len(parts) >= 3:
                cid = int(parts[0])
                nn_pass_test = float(parts[1])
                test_loss = float(parts[2])
                if cid not in data_test:
                    data_test[cid] = []
                data_test[cid].append((nn_pass_test, test_loss))
    for cid in data_test:
        data_test[cid].sort(key=lambda tup: tup[0])
    for cid in data_train:
        data_train[cid].sort(key=lambda tup: tup[0])
    return data_train, data_test

###############################################################################
# 2) Reading "server_accuracy.txt"
###############################################################################
def read_server_accuracy_file(file_path):
    """
    Expects lines of the form:
      GlobalRound   GlobalAccuracy
    Possibly with a header line.
    Returns a list of (round_number, accuracy).
    """
    data = []
    with open(file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if not parts[0].isdigit():
            continue
        rnd = int(parts[0])
        acc = float(parts[1])
        data.append((rnd, acc))
    data.sort(key=lambda t: t[0])
    return data

###############################################################################
# 3) Plot: Test Loss vs. NN Passes (single-row layout)
###############################################################################
def create_loss_figure_exp3(exp_title, gm_sass_dir, gm_sgd_dir, avg_sass_dir, avg_sgd_dir):
    """
    Creates a single-row subplot figure that plots test loss vs NN Passes for all four algorithms:
       - FedGM_SASS (red, marker "o")
       - FedGM_SGD  (blue, marker "s")
       - FedAvg_SASS (purple, marker "D")
       - FedAvg_SGD  (green, marker "^")
    """
    gm_sass_loss_file   = os.path.join(gm_sass_dir, "loss.txt")
    gm_sgd_loss_file    = os.path.join(gm_sgd_dir, "loss.txt")
    avg_sass_loss_file  = os.path.join(avg_sass_dir, "loss.txt")
    avg_sgd_loss_file   = os.path.join(avg_sgd_dir, "loss.txt")
    for fpath in [gm_sass_loss_file, gm_sgd_loss_file, avg_sass_loss_file, avg_sgd_loss_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find file: {fpath}")
    # For test loss, use the second dictionary from read_loss_file
    _, data_gm_sass = read_loss_file(gm_sass_loss_file)
    _, data_gm_sgd  = read_loss_file(gm_sgd_loss_file)
    _, data_avg_sass = read_loss_file(avg_sass_loss_file)
    _, data_avg_sgd  = read_loss_file(avg_sgd_loss_file)
    all_clients = sorted(list(set(data_gm_sass.keys()) | set(data_gm_sgd.keys()) | 
                               set(data_avg_sass.keys()) | set(data_avg_sgd.keys())))
    n_clients = len(all_clients)
    fig, axes = plt.subplots(nrows=1, ncols=n_clients, figsize=(6*n_clients, 6), squeeze=False)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    for j, cid in enumerate(all_clients):
        ax = axes[0, j]
        if cid in data_gm_sass:
            x, y = zip(*data_gm_sass[cid])
            ax.plot(x, y, label="FedGM_SASS", color="red", marker="o")
        if cid in data_gm_sgd:
            x, y = zip(*data_gm_sgd[cid])
            ax.plot(x, y, label="FedGM_SGD", color="blue", marker="s")
        if cid in data_avg_sass:
            x, y = zip(*data_avg_sass[cid])
            ax.plot(x, y, label="FedAvg_SASS", color="purple", marker="D")
        if cid in data_avg_sgd:
            x, y = zip(*data_avg_sgd[cid])
            ax.plot(x, y, label="FedAvg_SGD", color="green", marker="^")
        ax.set_title(f"Client {cid}")
        ax.set_xlabel("NN Passes")
        ax.set_ylabel("Test Loss")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.grid(True)
        ax.legend(fontsize=9)
    fig.suptitle(f"{exp_title}: Test Loss vs NN Passes (all 4 algorithms)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig

###############################################################################
# 3a) Plot: Training Loss vs. NN Passes (single-row layout)
###############################################################################
def create_training_loss_figure_exp3(exp_title, gm_sass_dir, gm_sgd_dir, avg_sass_dir, avg_sgd_dir):
    """
    Creates a single-row subplot figure that plots TRAIN loss vs NN Passes for all four algorithms:
       - FedGM_SASS (red, marker "o")
       - FedGM_SGD  (blue, marker "s")
       - FedAvg_SASS (purple, marker "D")
       - FedAvg_SGD  (green, marker "^")
    This function uses the training loss data (columns 1 and 2) from loss.txt.
    """
    data_train_gm_sass, _ = read_loss_file(os.path.join(gm_sass_dir, "loss.txt"))
    data_train_gm_sgd, _  = read_loss_file(os.path.join(gm_sgd_dir, "loss.txt"))
    data_train_avg_sass, _ = read_loss_file(os.path.join(avg_sass_dir, "loss.txt"))
    data_train_avg_sgd, _  = read_loss_file(os.path.join(avg_sgd_dir, "loss.txt"))
    all_clients = sorted(list(set(data_train_gm_sass.keys()) | set(data_train_gm_sgd.keys()) | 
                               set(data_train_avg_sass.keys()) | set(data_train_avg_sgd.keys())))
    n_clients = len(all_clients)
    fig, axes = plt.subplots(nrows=1, ncols=n_clients, figsize=(6*n_clients, 6), squeeze=False)
    fig.subplots_adjust(top=0.85, bottom=0.15)
    for j, cid in enumerate(all_clients):
        ax = axes[0, j]
        if cid in data_train_gm_sass:
            x, y = zip(*data_train_gm_sass[cid])
            ax.plot(x, y, label="FedGM_SASS", color="red", marker="o")
        if cid in data_train_gm_sgd:
            x, y = zip(*data_train_gm_sgd[cid])
            ax.plot(x, y, label="FedGM_SGD", color="blue", marker="s")
        if cid in data_train_avg_sass:
            x, y = zip(*data_train_avg_sass[cid])
            ax.plot(x, y, label="FedAvg_SASS", color="purple", marker="D")
        if cid in data_train_avg_sgd:
            x, y = zip(*data_train_avg_sgd[cid])
            ax.plot(x, y, label="FedAvg_SGD", color="green", marker="^")
        ax.set_title(f"Client {cid}")
        ax.set_xlabel("NN Passes")
        ax.set_ylabel("Train Loss")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.grid(True)
        ax.legend(fontsize=9)
    fig.suptitle(f"{exp_title}: Train Loss vs NN Passes (all 4 algorithms)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig

###############################################################################
# 4) New: Plot LR per Client vs Global Round (log-scale) from image files
###############################################################################
def create_lr_figure(exp_title, gm_sass_dir, avg_sass_dir):
    """
    Creates a figure that shows the lr_per_client_per_global_round_log.png images 
    from the FedGM_SASS and FedAvg_SASS folders side by side.
    """
    gm_lr_file = os.path.join(gm_sass_dir, "lr_per_client_per_global_round_log.png")
    avg_lr_file = os.path.join(avg_sass_dir, "lr_per_client_per_global_round_log.png")
    if not os.path.exists(gm_lr_file):
         raise FileNotFoundError(f"Could not find file: {gm_lr_file}")
    if not os.path.exists(avg_lr_file):
         raise FileNotFoundError(f"Could not find file: {avg_lr_file}")
    img_gm = mpimg.imread(gm_lr_file)
    img_avg = mpimg.imread(avg_lr_file)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_gm)
    axes[0].set_title("FedGM_SASS LR (log-scale)")
    axes[0].axis('off')
    axes[1].imshow(img_avg)
    axes[1].set_title("FedAvg_SASS LR (log-scale)")
    axes[1].axis('off')
    fig.suptitle(f"{exp_title}: LR per Client vs Global Round (log-scale)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

###############################################################################
# 5) Plot: Server Accuracy vs Global Round with external text
###############################################################################
def create_accuracy_figure_exp3(exp_title, gm_sass_dir, gm_sgd_dir, avg_sass_dir, avg_sgd_dir):
    """
    Creates a figure for server accuracy vs global round for experiments 3 and 4 (four algorithms).
    The final accuracy numbers are displayed in a dedicated area above the main plot.
    """
    gm_sass_acc_file   = os.path.join(gm_sass_dir, "server_accuracy.txt")
    gm_sgd_acc_file    = os.path.join(gm_sgd_dir, "server_accuracy.txt")
    avg_sass_acc_file  = os.path.join(avg_sass_dir, "server_accuracy.txt")
    avg_sgd_acc_file   = os.path.join(avg_sgd_dir, "server_accuracy.txt")
    for fpath in [gm_sass_acc_file, gm_sgd_acc_file, avg_sass_acc_file, avg_sgd_acc_file]:
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Could not find server_accuracy.txt: {fpath}")
    data_gm_sass   = read_server_accuracy_file(gm_sass_acc_file)
    data_gm_sgd    = read_server_accuracy_file(gm_sgd_acc_file)
    data_avg_sass  = read_server_accuracy_file(avg_sass_acc_file)
    data_avg_sgd   = read_server_accuracy_file(avg_sgd_acc_file)

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[0.2, 0.8])
    
    ax_text = fig.add_subplot(gs[0])
    ax_text.axis('off')
    lines_text = []
    if data_gm_sass:
        x, y = zip(*data_gm_sass)
        lines_text.append(f"FedGM_SASS final acc: {y[-1]:.2f}%")
    if data_gm_sgd:
        x, y = zip(*data_gm_sgd)
        lines_text.append(f"FedGM_SGD final acc : {y[-1]:.2f}%")
    if data_avg_sass:
        x, y = zip(*data_avg_sass)
        lines_text.append(f"FedAvg_SASS final acc: {y[-1]:.2f}%")
    if data_avg_sgd:
        x, y = zip(*data_avg_sgd)
        lines_text.append(f"FedAvg_SGD final acc : {y[-1]:.2f}%")
    text_block = "\n".join(lines_text)
    ax_text.text(0.01, 0.5, text_block, ha='left', va='center', fontsize=11,
                 bbox=dict(facecolor='white', alpha=0.7))
    
    ax = fig.add_subplot(gs[1])
    if data_gm_sass:
        x, y = zip(*data_gm_sass)
        ax.plot(x, y, color='red', marker='o', label="FedGM_SASS")
    if data_gm_sgd:
        x, y = zip(*data_gm_sgd)
        ax.plot(x, y, color='blue', marker='s', label="FedGM_SGD")
    if data_avg_sass:
        x, y = zip(*data_avg_sass)
        ax.plot(x, y, color='purple', marker='D', label="FedAvg_SASS")
    if data_avg_sgd:
        x, y = zip(*data_avg_sgd)
        ax.plot(x, y, color='green', marker='^', label="FedAvg_SGD")
    ax.set_xlabel("Global Round")
    ax.set_ylabel("Server Accuracy (%)")
    ax.grid(True)
    ax.legend()
    
    fig.suptitle(f"{exp_title} - Server Accuracy vs Global Round", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

###############################################################################
# 6) MAIN
###############################################################################
def main():
    output_pdf = "/home/local/ASURITE/yzeng88/fedSASS/A_experiment_output_cifar10/compare_all_experiments.pdf"
    pp = PdfPages(output_pdf)

    # (Experiment 1 is not used for CIFAR-10)

    # Experiment 2: dirichlet non-iid (4 algorithms)
    exp2_title = "Experiment 2: dirichlet non-iid"
    # Page A: Test Loss vs NN Passes
    fig2 = create_loss_figure_exp3(exp_title=exp2_title,
                                   gm_sass_dir=fedgm_sass_dir_exp2,
                                   gm_sgd_dir=fedgm_sgd_dir_exp2,
                                   avg_sass_dir=fedavg_sass_dir_exp2,
                                   avg_sgd_dir=fedavg_sgd_dir_exp2)
    pp.savefig(fig2)
    plt.close(fig2)
    # Extra Page: Training Loss vs NN Passes
    fig2_train = create_training_loss_figure_exp3(exp_title=exp2_title,
                                                  gm_sass_dir=fedgm_sass_dir_exp2,
                                                  gm_sgd_dir=fedgm_sgd_dir_exp2,
                                                  avg_sass_dir=fedavg_sass_dir_exp2,
                                                  avg_sgd_dir=fedavg_sgd_dir_exp2)
    pp.savefig(fig2_train)
    plt.close(fig2_train)
    # Page B: Server Accuracy vs Global Round (4 algorithms)
    fig2_acc = create_accuracy_figure_exp3(exp_title=exp2_title,
                                           gm_sass_dir=fedgm_sass_dir_exp2,
                                           gm_sgd_dir=fedgm_sgd_dir_exp2,
                                           avg_sass_dir=fedavg_sass_dir_exp2,
                                           avg_sgd_dir=fedavg_sgd_dir_exp2)
    pp.savefig(fig2_acc)
    plt.close(fig2_acc)
    # Page C: LR per Client vs Global Round (log-scale)
    fig2_lr = create_lr_figure(exp_title=exp2_title,
                               gm_sass_dir=fedgm_sass_dir_exp2,
                               avg_sass_dir=fedavg_sass_dir_exp2)
    pp.savefig(fig2_lr)
    plt.close(fig2_lr)

    # Experiment 3: case2 (4 algorithms)
    exp3_title = "Experiment 3: case2"
    # Page A: Test Loss vs NN Passes
    fig3 = create_loss_figure_exp3(exp_title=exp3_title,
                                   gm_sass_dir=fedgm_sass_dir_exp3,
                                   gm_sgd_dir=fedgm_sgd_dir_exp3,
                                   avg_sass_dir=fedavg_sass_dir_exp3,
                                   avg_sgd_dir=fedavg_sgd_dir_exp3)
    pp.savefig(fig3)
    plt.close(fig3)
    # Extra Page: Training Loss vs NN Passes
    fig3_train = create_training_loss_figure_exp3(exp_title=exp3_title,
                                                  gm_sass_dir=fedgm_sass_dir_exp3,
                                                  gm_sgd_dir=fedgm_sgd_dir_exp3,
                                                  avg_sass_dir=fedavg_sass_dir_exp3,
                                                  avg_sgd_dir=fedavg_sgd_dir_exp3)
    pp.savefig(fig3_train)
    plt.close(fig3_train)
    # Page B: Server Accuracy vs Global Round with external text block
    fig3_acc = create_accuracy_figure_exp3(exp_title=exp3_title,
                                           gm_sass_dir=fedgm_sass_dir_exp3,
                                           gm_sgd_dir=fedgm_sgd_dir_exp3,
                                           avg_sass_dir=fedavg_sass_dir_exp3,
                                           avg_sgd_dir=fedavg_sgd_dir_exp3)
    pp.savefig(fig3_acc)
    plt.close(fig3_acc)
    # Page C: LR per Client vs Global Round (log-scale)
    fig3_lr = create_lr_figure(exp_title=exp3_title,
                               gm_sass_dir=fedgm_sass_dir_exp3,
                               avg_sass_dir=fedavg_sass_dir_exp3)
    pp.savefig(fig3_lr)
    plt.close(fig3_lr)

    # Experiment 4: case3 (4 algorithms)
    exp4_title = "Experiment 4: case3"
    # Page A: Test Loss vs NN Passes
    fig4 = create_loss_figure_exp3(exp_title=exp4_title,
                                   gm_sass_dir=fedgm_sass_dir_exp4,
                                   gm_sgd_dir=fedgm_sgd_dir_exp4,
                                   avg_sass_dir=fedavg_sass_dir_exp4,
                                   avg_sgd_dir=fedavg_sgd_dir_exp4)
    pp.savefig(fig4)
    plt.close(fig4)
    # Extra Page: Training Loss vs NN Passes
    fig4_train = create_training_loss_figure_exp3(exp_title=exp4_title,
                                                  gm_sass_dir=fedgm_sass_dir_exp4,
                                                  gm_sgd_dir=fedgm_sgd_dir_exp4,
                                                  avg_sass_dir=fedavg_sass_dir_exp4,
                                                  avg_sgd_dir=fedavg_sgd_dir_exp4)
    pp.savefig(fig4_train)
    plt.close(fig4_train)
    # Page B: Server Accuracy vs Global Round with external text block
    fig4_acc = create_accuracy_figure_exp3(exp_title=exp4_title,
                                           gm_sass_dir=fedgm_sass_dir_exp4,
                                           gm_sgd_dir=fedgm_sgd_dir_exp4,
                                           avg_sass_dir=fedavg_sass_dir_exp4,
                                           avg_sgd_dir=fedavg_sgd_dir_exp4)
    pp.savefig(fig4_acc)
    plt.close(fig4_acc)
    # Page C: LR per Client vs Global Round (log-scale)
    fig4_lr = create_lr_figure(exp_title=exp4_title,
                               gm_sass_dir=fedgm_sass_dir_exp4,
                               avg_sass_dir=fedavg_sass_dir_exp4)
    pp.savefig(fig4_lr)
    plt.close(fig4_lr)

    pp.close()
    print(f"\nCombined PDF saved to: {output_pdf}\n")

if __name__ == "__main__":
    main()
