import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def generate_comparison_report(experiments, output_pdf):
    """
    Generate a multi-page PDF report comparing FedGM vs. FedAvg side by side.

    Parameters:
    -----------
    experiments : list of dict
        Each dict should have:
            {
                "title": str,          # Experiment title
                "fedgm_dir": str,      # Path to FedGM figures
                "fedavg_dir": str,     # Path to FedAvg figures
                "description": str,    # Text/description for this experiment
            }
    output_pdf : str
        Path to the output PDF file that will contain the entire report.
    """

    # List of (filename, label_for_the_row) in the order we want to display them
    figure_info = [
        ("client_data_dist.png", "Data Distribution"),
        ("accuracy_clients_and_global.png", "Accuracy"),
        ("lr_per_client_per_global_epoch.png", "Learning Rate"),
        ("lr_per_client_per_global_epoch_log.png", "Log (LR)"),
        ("training_loss_per_client.png", "Training Loss"),
        ("grad_norm_per_client_per_global_epoch.png", "Gradient Norm"),
    ]

    # Create a PDF file to write to
    with PdfPages(output_pdf) as pdf:
        for exp in experiments:
            # Create a figure with as many rows as the number of figure types
            # and 2 columns (FedGM on the left, FedAvg on the right).
            nrows = len(figure_info)
            fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(14, 5 * nrows))

            # If there's only one row, axes is 1D; otherwise it's 2D. Let's ensure 2D always.
            if nrows == 1:
                axes = [axes]  # wrap in list so we can index as axes[row][col]

            # Add a title at the top
            fig.suptitle(exp["title"], fontsize=16, fontweight="bold", y=0.97)

            # Optionally add the experiment's descriptive text below the title:
            # (We can place it with fig.text(...) or we can let suptitle do everything.)
            # We'll place the description near the top left, just below the suptitle.
            fig.text(0.05, 0.93, exp["description"], fontsize=14, va="top")

            # Loop over each row (figure type)
            for row_idx, (filename, label) in enumerate(figure_info):
                # Left column: FedGM
                left_ax = axes[row_idx][0]
                gm_path = os.path.join(exp["fedgm_dir"], filename)
                if os.path.exists(gm_path):
                    img_gm = mpimg.imread(gm_path)
                    left_ax.imshow(img_gm)
                    left_ax.set_title(f"FedGM - {label}", fontsize=11)
                else:
                    left_ax.text(0.5, 0.5, f"Missing:\n{gm_path}", ha="center", va="center")
                    left_ax.set_title(f"FedGM - {label} (Not Found)", fontsize=11)
                left_ax.axis("off")

                # Right column: FedAvg
                right_ax = axes[row_idx][1]
                avg_path = os.path.join(exp["fedavg_dir"], filename)
                if os.path.exists(avg_path):
                    img_avg = mpimg.imread(avg_path)
                    right_ax.imshow(img_avg)
                    right_ax.set_title(f"FedAvg - {label}", fontsize=11)
                else:
                    right_ax.text(0.5, 0.5, f"Missing:\n{avg_path}", ha="center", va="center")
                    right_ax.set_title(f"FedAvg - {label} (Not Found)", fontsize=11)
                right_ax.axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.92])  # leave room at top for suptitle/description
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Report saved to: {output_pdf}")


if __name__ == "__main__":
    # EXAMPLE USAGE:
    experiments = [
        {
            "title": "Experiment 1 (Dirichlet alpha=0.1, 5 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_Extremenoniid_Dirichlet[80,0.1]_5_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_extremeNoniid_Dirichlet[8,0.1]__5_Client_50_epoch",
            "description": (
                "In this experiment, we used 5 clients with Dirichlet partition (alpha=0.1). \n"
                "The Accuracy of FedGM reached [91.26, 86.91, 91.57, 93.79, 91.91] for each client.\n"
                "The Accuracy of FedAvg reached [93.76, 87.05, 92.92, 88.42, 90.88] for each client."
            )
        },
        {
            "title": "Experiment 2 (Mild nonIID alpha=1, 5 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_Mildnoniid_Dirichlet[80,1]_5_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_mildNoniid_Dirichlet[8,1]__5_Client_50_epoch",
            "description": (
                "In this experiment, data is mild non IID across 5 clients. \n"
                "The Accuracy of FedGM reached [98.91, 98.81, 98.83, 98.91, 98.99 ] for each client.\n"
                "The Accuracy of FedAvg reached [98.99, 99.02, 98.70, 98.91, 98.93 ] for each client."
            )
        },
        {
            "title": "Experiment 3 (IID alpha=1, 5 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_iid_5_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_iid_5_Client_50_epoch",
            "description": (
                "In this experiment, data is IID across 5 clients. \n"
                "The Accuracy of FedGM reached [99.02, 99.07, 99.01, 99.08, 99.12 ] for each client.\n"
                "The Accuracy of FedAvg reached [99.22, 99.19, 99.24, 99.25, 99.17 ] for each client."
            )
        },
        {
            "title": "Experiment 4 (Extreme nonIID alpha=0.1, 10 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_Extremenoniid_Dirichlet[80,0.1]_10_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_ExtremeNoniid_Dirichlet[8,0.1]__10_Client_50_epoch",
            "description": (
                "In this experiment, data is extreme non IID across 10 clients.\n "
                "The Accuracy of FedGM reached [87.54, 95.99, 95.14, 91.45, 93.41, 88.45, 95.29, 97.55, 90.87, 93.10 ] for each client.\n"
                "The Accuracy of FedAvg reached [91.89, 95.84, 92.66, 94.90, 95.62, 85.69, 96.12, 95.69, 83.98, 96.90 ] for each client."
            )
        },
        {
            "title": "Experiment 5 (Mild nonIID alpha=1, 10 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_Mildnoniid_Dirichlet[80,1]_10_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_mildNoniid_Dirichlet[8,1]__10_Client_50_epoch",
            "description": (
                "In this experiment, data is mild non IID across 10 clients. \n"
                "The Accuracy of FedGM:[98.83, 98.56, 98.86, 98.93, 98.78, 98.66, 98.62, 98.44, 98.67, 98.95 ] \n"
                "The Accuracy of FedAvg:[98.76, 98.85, 98.71, 98.84, 98.98, 98.89, 98.66, 98.03, 98.90, 99.01 ]"
            )
        },
                {
            "title": "Experiment 6 (IID, 10 clients, 50 epoch)",
            "fedgm_dir": "/home/local/ASURITE/yzeng88/fedSASS/FedGM_MNIST_iid_10_Client__50_epoch",
            "fedavg_dir": "/home/local/ASURITE/yzeng88/fedSASS/A_sass_FedAvg_MNIST_iid_10_Client_50_epoch",
            "description": (
                "In this experiment, data is IID across 10 clients. \n"
                "We see how FedGM compares to FedAvg in a more balanced setting."
                "The Accuracy of FedGM: [99.01, 98.86, 98.86, 98.92, 98.90, 98.97, 98.98, 98.77, 98.94, 98.93 ] \n"
                "The Accuracy of FedAvg: [99.01, 98.92, 98.92, 99.07, 99.04, 99.05, 98.73, 99.00, 99.00, 99.00 ]"
            )
        },

        # add more experiments as needed... 10 clients
    ]

    # Output PDF path
    output_file = "Comparison_Report_FedGM_vs_FedAvg.pdf"

    # Generate the report
    generate_comparison_report(experiments, output_file)
