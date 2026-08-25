import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch


def simple_plot(data, device, filepath):
    epochs = range(1, len(data.get("train_loss")) + 1)
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, data.get("train_loss"), marker="o", color="teal", label="Train Loss")

    if data.get("val_loss"):
        plt.plot(epochs, data.get("val_loss"), marker="o", color="orange", label="Val Loss", linestyle="--")
    if data.get("early_stop"):
        plt.axvline(x=data["early_stop"], color="crimson", linestyle="--", label="Early Stop")
    if data.get("checkpoints"):
        for i, point in enumerate(data["checkpoints"]):
            label = "Loss cp" if i == 0 else None
            plt.scatter(point, data["val_loss"][point -1], color="crimson", s=50, label=label, zorder=5)

    plt.title("Loss per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, data.get("train_acc"), marker="o", color="green", label="Train Acc")

    if data.get("val_acc"):
        plt.plot(epochs, data.get("val_acc"), marker="o", color="gold", label="Val Acc", linestyle="--")
    if data.get("early_stop"):
        plt.axvline(x=data["early_stop"], color="crimson", linestyle="--", label="Early Stop")
    if data.get("checkpoints"):
        for i, point in enumerate(data["checkpoints"]):
            label = "Acc cp" if i == 0 else None
            plt.scatter(point, data["val_acc"][point -1], color="crimson", s=50, label=label, zorder=5)

    plt.title("Accuracy per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()

    # Utilisation
    if data.get("utils"):
        plt.subplot(1, 3, 3)
        plt.plot(epochs, data.get("utils"), marker="o", color="purple", label=f"{device} Util")

        if data.get("early_stop"):
            plt.axvline(x=data["early_stop"], color="crimson", linestyle="--", label="Early Stop")

        plt.title(f"Avg {device} Utilisation")
        plt.xlabel("Epoch")
        plt.ylabel("Utilisation (%)")
        plt.ylim(0, 100)
        plt.legend()

    for ax in plt.gcf().get_axes():
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    logging.info(f"Metric visualisation saved to {filepath}")


def detailed_plot(data, filepath):
    """Plot comprehensive training metrics in 2x3 grid"""
    epochs = range(1, len(data.get("train_loss")) + 1)
    fig = plt.figure(figsize=(18, 12))

    def _plot_metric(subplot_idx, train_data, val_data, title, ylabel, 
                   color_train="teal", color_val="orange", ylim=None):
        plt.subplot(2, 3, subplot_idx)
        plt.plot(epochs, train_data, marker="o", color=color_train, 
                label=f"Train {title}", linewidth=2, markersize=4)

        if val_data:
            plt.plot(epochs, val_data, marker="s", color=color_val, 
                    label=f"Val {title}", linestyle="--", linewidth=2, markersize=4)

        if data.get("early_stop", None):
            plt.axvline(x=data["early_stop"], color="crimson", 
                       linestyle=":", linewidth=2, label="Early Stop", alpha=0.7)

        if data.get("checkpoints", None) and val_data:
            for i, cp_epoch in enumerate(data["checkpoints"]):
                label = f"{title} Checkpoint" if i == 0 else None
                plt.scatter(cp_epoch, val_data[cp_epoch - 1], 
                           color="crimson", s=100, marker="*", 
                           label=label, zorder=5, edgecolors="black", linewidths=1)

        plt.title(f"{title} per Epoch", fontsize=12, fontweight='bold')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel(ylabel, fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        if ylim:
            plt.ylim(ylim)
        plt.legend(fontsize=9)

    # Row 1: Loss, Accuracy, F1
    _plot_metric(1, data.get("train_loss"), data.get("val_loss"),
               "Loss", "Loss", color_train="teal", color_val="orange")

    _plot_metric(2, data.get("train_acc"), data.get("val_acc"),
               "Accuracy", "Accuracy (%)", color_train="green", color_val="gold",
               ylim=(0, 100))

    if data["train_f1"] or data["val_f1"]:
        _plot_metric(3, data.get("train_f1"), data.get("val_f1"),
                   "F1 Score", "F1 Score", color_train="royalblue", color_val="coral",
                   ylim=(0, 1))
    else:
        # Empty subplot if no F1
        plt.subplot(2, 3, 3)
        plt.axis('off')

    # Row 2: Memory, CPU/GPU Util
    # Memory Util
    if data["mem_utils"]:
        plt.subplot(2, 3, 4)
        memory_gb = [m / 1024 for m in data.get("mem_utils")]
        plt.plot(epochs, memory_gb, marker="o", color="purple",
                label="Mem Util", linewidth=2, markersize=4)

        if data.get("early_stop", None):
            plt.axvline(x=data["early_stop"], color="crimson",
                       linestyle=":", linewidth=2, label="Early Stop", alpha=0.7)

        plt.title("Memory Util per Epoch", fontsize=12, fontweight='bold')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Memory (GB)", fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=9)

    # CPU and GPU Util
    if data["cpu_utils"] or data["gpu_utils"]:
        plt.subplot(2, 3, 5)

        if data["cpu_utils"]:
            plt.plot(epochs, data.get("cpu_utils"), marker="o", color="orange",
                    label="CPU Util", linewidth=2, markersize=4)

        if data["gpu_utils"]:
            plt.plot(epochs, data.get("gpu_utils"), marker="s", color="darkgreen",
                    label="GPU Util", linestyle="--", linewidth=2, markersize=4)

        if data.get("early_stop", None):
            plt.axvline(x=data["early_stop"], color="crimson",
                       linestyle=":", linewidth=2, label="Early Stop", alpha=0.7)

        plt.title("Device Util per Epoch", fontsize=12, fontweight='bold')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Util (%)", fontsize=10)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=9)

    # Batch wait time
    if data["batch_wait_times"]:
        plt.subplot(2, 3, 6)

        plt.plot(epochs, data.get("batch_wait_times"), marker="o", color="orange",
                 label="Batch Wait", linewidth=2, markersize=4)

        if data.get("early_stop", None):
            plt.axvline(x=data["early_stop"], color="crimson",
                       linestyle=":", linewidth=2, label="Early Stop", alpha=0.7)

        plt.title("Batch Loading Time", fontsize=12, fontweight='bold')
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Wait Time (ms)", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)

    else:
        plt.subplot(2, 3, 6)
        plt.axis('off')

    for ax in plt.gcf().get_axes():
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Training visualization saved to {filepath}")


def show_predictions(model, data, device, filepath, num_samples=16):
    model.eval()

    images_list = []
    labels_list = []
    predictions_list = []
    confidences_list = []

    with torch.no_grad():
        for images, labels in data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)

            # Collect samples
            for i in range(len(images)):
                if len(images_list) >= num_samples:
                    break

                images_list.append(images[i].cpu())
                labels_list.append(labels[i].cpu().item())
                predictions_list.append(predictions[i].cpu().item())
                confidences_list.append(confidences[i].cpu().item())

            if len(images_list) >= num_samples:
                break

    grid_size = int(np.ceil(np.sqrt(num_samples)))
    _, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    class_names = ['Real', 'Fake']

    for idx, ax in enumerate(axes):
        if idx < len(images_list):
            # Denormalize image for display
            img = images_list[idx]
            img = img.permute(1, 2, 0)  # CHW -> HWC
            img = torch.clamp(img, 0, 1)
            ax.imshow(img)

            # Get prediction info
            true_label = labels_list[idx]
            pred_label = predictions_list[idx]
            confidence = confidences_list[idx]

            correct = (true_label == pred_label)
            color = 'green' if correct else 'red'

            # Title with prediction info
            title = f"True: {class_names[true_label]}\n"
            title += f"Pred: {class_names[pred_label]} ({confidence:.2%})"

            ax.set_title(title, color=color, fontsize=10, weight='bold')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Predictions visualisation saved to {filepath}")
