#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = "/home/local/ASURITE/yzeng88/fedSASS/Jan29_output"

# Pairs of experiments with the "same setting":
EXPERIMENT_PAIRS = [
    ("fedavg_dirichlet_1clients_50rounds", "fedgm_dirichlet_1clients_50rounds"),
    ("fedavg_dirichlet_2clients_100rounds", "fedgm_dirichlet_2clients_100rounds"),
    ("fedavg_dirichlet_3clients_200rounds", "fedgm_dirichlet_3clients_200rounds"),
    ("fedavg_dirichlet_5clients_300rounds", "fedgm_dirichlet_5clients_300rounds"),
    ("fedavg_dirichlet_8clients_300rounds", "fedgm_dirichlet_8clients_300rounds"),
    ("fedavg_iid_5clients_300rounds",      "fedgm_iid_5clients_300rounds"),
]

# Added "data_dist_B.png" here, so we now compare it as well.
FIGURE_FILENAMES = [
    "accuracy_clients_and_global.png",
    "data_dist.png",
    "data_dist_B.png",  # <--- Newly added
    "grad_norm_per_client_per_global_round.png",
    "lr_per_client_per_global_round_log.png",
    "lr_per_client_per_global_round.png",
    "training_loss_per_client.png",
]

def parse_result_txt(result_txt_path):
    """
    Parses 'result.txt' lines, which might look like:
      GlobalRnd  ClientID  LocEpoch  Accuracy  LR  Loss  GradNorm
      1          0         1         97.9400   ... ...
      ...
    We return {client_id -> final_accuracy}, using the last line for each client.
    """
    client_final_acc = {}
    if not os.path.isfile(result_txt_path):
        return client_final_acc

    with open(result_txt_path, 'r') as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            # Skip header line if it starts with "GlobalRnd"
            if not header_skipped and ("GlobalRnd" in line and "ClientID" in line):
                header_skipped = True
                continue
            if not line:
                continue

            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                client_id = int(parts[1])
                accuracy  = float(parts[3])
            except ValueError:
                continue
            # Overwrite with the latest accuracy for that client
            client_final_acc[client_id] = accuracy

    return client_final_acc

def read_server_accuracy(server_acc_path):
    """
    Reads server_accuracy.txt lines:
       GlobalRound  GlobalAccuracy
       1            97.3900
       2            98.0400
       ...
    Returns (rounds, accuracies).
    """
    rounds, accs = [], []
    if not os.path.isfile(server_acc_path):
        return rounds, accs

    with open(server_acc_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("GlobalRound"):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    r = int(parts[0])
                    a = float(parts[1])
                    rounds.append(r)
                    accs.append(a)
                except ValueError:
                    pass
    return rounds, accs

def main():
    pdf_path = os.path.join(BASE_DIR, "fedavg_vs_fedgm_report.pdf")
    with PdfPages(pdf_path) as pdf:
        
        for fedavg_exp, fedgm_exp in EXPERIMENT_PAIRS:
            fedavg_path = os.path.join(BASE_DIR, fedavg_exp)
            fedgm_path  = os.path.join(BASE_DIR, fedgm_exp)

            # Final client accuracies (from result.txt)
            fedavg_client_accs = parse_result_txt(os.path.join(fedavg_path, "result.txt"))
            fedgm_client_accs  = parse_result_txt(os.path.join(fedgm_path, "result.txt"))

            # Server accuracy for final global accuracy
            fedavg_rounds, fedavg_accs = read_server_accuracy(
                os.path.join(fedavg_path, "server_accuracy.txt"))
            fedgm_rounds,  fedgm_accs  = read_server_accuracy(
                os.path.join(fedgm_path, "server_accuracy.txt"))

            fedavg_final_global = fedavg_accs[-1] if fedavg_accs else None
            fedgm_final_global  = fedgm_accs[-1]  if fedgm_accs  else None

            ################################################################
            # 1) Page for "accuracy_clients_and_global.png" with text block
            ################################################################
            acc_fig_name = "accuracy_clients_and_global.png"
            fedavg_acc_fig = os.path.join(fedavg_path, acc_fig_name)
            fedgm_acc_fig  = os.path.join(fedgm_path,  acc_fig_name)

            fig = plt.figure(figsize=(10,7))
            fig.suptitle(f"{fedavg_exp} vs. {fedgm_exp} | {acc_fig_name}", fontsize=10)

            ax_text = fig.add_subplot(2,1,1)
            ax_text.axis('off')

            fedavg_text = ["FedAvg final client accuracies:"]
            for cid in sorted(fedavg_client_accs.keys()):
                fedavg_text.append(f"  Client {cid}: {fedavg_client_accs[cid]:.4f}")
            if fedavg_final_global is not None:
                fedavg_text.append(f"  Final global accuracy: {fedavg_final_global:.4f}")

            fedgm_text = ["FedGM final client accuracies:"]
            for cid in sorted(fedgm_client_accs.keys()):
                fedgm_text.append(f"  Client {cid}: {fedgm_client_accs[cid]:.4f}")
            if fedgm_final_global is not None:
                fedgm_text.append(f"  Final global accuracy: {fedgm_final_global:.4f}")

            combined_text = "\n".join(fedavg_text) + "\n\n" + "\n".join(fedgm_text)
            ax_text.text(0.01, 0.95, combined_text, fontsize=9,
                         va='top', ha='left', transform=ax_text.transAxes)

            ax_left  = fig.add_subplot(2,2,3)
            ax_right = fig.add_subplot(2,2,4)

            if os.path.isfile(fedavg_acc_fig):
                img_left = plt.imread(fedavg_acc_fig)
                ax_left.imshow(img_left)
                ax_left.set_title(fedavg_exp, fontsize=9)
            ax_left.axis('off')

            if os.path.isfile(fedgm_acc_fig):
                img_right = plt.imread(fedgm_acc_fig)
                ax_right.imshow(img_right)
                ax_right.set_title(fedgm_exp, fontsize=9)
            ax_right.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            ################################################################
            # 2) For other figures, side-by-side pages (FedAvg left, FedGM right)
            ################################################################
            other_figs = [f for f in FIGURE_FILENAMES if f != acc_fig_name]
            for fig_name in other_figs:
                fedavg_fig = os.path.join(fedavg_path, fig_name)
                fedgm_fig  = os.path.join(fedgm_path,  fig_name)

                fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
                fig.suptitle(f"{fedavg_exp} vs. {fedgm_exp} | {fig_name}", fontsize=10)

                if os.path.isfile(fedavg_fig):
                    img_left = plt.imread(fedavg_fig)
                    axes[0].imshow(img_left)
                    axes[0].set_title(fedavg_exp, fontsize=9)
                axes[0].axis('off')

                if os.path.isfile(fedgm_fig):
                    img_right = plt.imread(fedgm_fig)
                    axes[1].imshow(img_right)
                    axes[1].set_title(fedgm_exp, fontsize=9)
                axes[1].axis('off')

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            ################################################################
            # 3) FedAvg vs FedGM server_accuracy.txt line chart
            ################################################################
            if (fedavg_rounds and fedavg_accs) or (fedgm_rounds and fedgm_accs):
                fig, ax = plt.subplots(figsize=(6,4))
                fig.suptitle(f"{fedavg_exp} vs. {fedgm_exp} | Server Accuracy", fontsize=10)

                if fedavg_rounds and fedavg_accs:
                    ax.plot(fedavg_rounds, fedavg_accs, label=f"{fedavg_exp}")
                if fedgm_rounds and fedgm_accs:
                    ax.plot(fedgm_rounds,  fedgm_accs,  label=f"{fedgm_exp}")

                ax.set_xlabel("Global Round")
                ax.set_ylabel("Global Accuracy")
                ax.legend()
                ax.grid(True)

                pdf.savefig(fig)
                plt.close(fig)

    print(f"Report generated at: {pdf_path}")

if __name__ == "__main__":
    main()
