#!/usr/bin/python3
import logging
import torch
import time

from deepfake_detection import utils, eval
from sklearn.metrics import f1_score


def lr_warmup(epoch, base_lr, warmup_epochs, total_epochs):
    import math

    if epoch < warmup_epochs:
        # Linear warmup
        lr = base_lr * (epoch + 1) / warmup_epochs
        logging.info(f"Applying linear warmup. LR: {lr:.8f}")
        return lr
            
    # Cosine decay
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    logging.info(f"Applying cosine decay. LR: {lr:.8f}")
    return lr 


def train(device, model, train_data, val_data, epochs, learn_rate, patience, filepath, warmup):
    '''Train and validate model.

    Args:
        device (obj): torch device
        model (obj): model
        train_data (_type_): torch train dataloader
        val_data (_type_): torch validation dataloader
        epochs (_type_): number of epochs
        learn_rate (_type_): learning rate for optimiser
        filepath (str): output filepath
        warmup (int): number of epochs to apply linear warmup
    '''
    train_metrics = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
        "mem_utils": [],
        "cpu_utils": [],
        "gpu_utils": [],
        "batch_wait_times": [],
        "checkpoints": []
    }
    best_accuracy = 0.0
    best_loss = float('inf')
    degradation = 0

    # Create optimiser and loss function
    optimiser = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train_samples = 0

        # For tracking epoch utilisation
        epoch_cpu_utils = []
        epoch_gpu_utils = []
        epoch_mem_utils = []
        epoch_wait_times = []

        # For F1 calculation
        all_train_preds = []
        all_train_labels = []

        # Update learning rate for this epoch
        epoch_lr = lr_warmup(epoch, learn_rate, warmup, epochs)

        for param_group in optimiser.param_groups:
            param_group["lr"] = epoch_lr

        # Track batch loading time
        iter_start = time.time()

        for train_images, train_labels in train_data:
            # Measure batch wait time
            batch_ready_time = time.time()
            wait_time = batch_ready_time - iter_start
            epoch_wait_times.append(wait_time * 1000)

            # Train
            train_images, train_labels = train_images.to(device), train_labels.to(device)
            outputs = model(train_images)

            optimiser.zero_grad()
            train_loss = loss_fn(outputs, train_labels)
            train_loss.backward()
            optimiser.step()

            total_train_loss += train_loss.item()

            train_preds = torch.argmax(outputs, axis=1)
            correct_train += (train_preds == train_labels).sum().item()
            total_train_samples += train_labels.size(0)

            # Collect for F1 score
            all_train_preds.extend(train_preds.cpu().numpy())
            all_train_labels.extend(train_labels.cpu().numpy())

            # Collect epoch metrics
            epoch_utils = utils.get_system_util(device)
            epoch_cpu_utils.append(epoch_utils["cpu"])

            if epoch_utils["gpu"] is not None:
                epoch_gpu_utils.append(epoch_utils["gpu"])

            epoch_mem_utils.append(utils.get_memory_util(device))

            # Start timer for next batch
            iter_start = time.time()

        # Calculate training metrics
        train_acc = 100 * correct_train / total_train_samples
        train_loss = total_train_loss / len(train_data)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='binary')

        train_metrics["train_acc"].append(train_acc)
        train_metrics["train_loss"].append(train_loss)
        train_metrics["train_f1"].append(train_f1)

        # Validation with F1 score
        val_metrics = eval.eval_model(model, device, val_data, loss_fn)

        train_metrics["val_acc"].append(val_metrics["eval_acc"])
        train_metrics["val_loss"].append(val_metrics["eval_loss"])
        train_metrics["val_f1"].append(val_metrics["eval_f1"])

        # Combine train and val CPU util
        combined_cpu = epoch_cpu_utils + val_metrics["cpu_utils"]
        avg_cpu_util = sum(combined_cpu) / len(combined_cpu) if len(combined_cpu) > 0 else 0
        train_metrics["cpu_utils"].append(avg_cpu_util)

        # Combine train and val GPU util
        if epoch_gpu_utils or val_metrics["gpu_utils"]:
            combined_gpu = epoch_gpu_utils + val_metrics["gpu_utils"]
            avg_gpu_util = sum(combined_gpu) / len(combined_gpu) if len(combined_gpu) > 0 else 0
            train_metrics["gpu_utils"].append(avg_gpu_util)

        # Combine train and val memory util
        combined_mem = epoch_mem_utils + val_metrics["mem_utils"]
        avg_mem_util = sum(combined_mem) / len(combined_mem) if len(combined_mem) > 0 else 0
        train_metrics["mem_utils"].append(avg_mem_util)

        # Avg batch wait time for this epoch
        combined_wait = epoch_wait_times + val_metrics.get("batch_wait_times", [])
        avg_wait_time = sum(combined_wait) / len(combined_wait) if len(combined_wait) > 0 else 0
        train_metrics["batch_wait_times"].append(avg_wait_time)

        log_msg = (
            f"Epoch: {epoch+1}  "
            f"Train Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.2f}%  "
            f"Train F1: {train_f1:.4f}  "
            f"Val Loss: {val_metrics['eval_loss']:.4f}  "
            f"Val Acc: {val_metrics['eval_acc']:.2f}%  "
            f"Val F1: {val_metrics['eval_f1']:.4f}  "
            f"Mem: {avg_mem_util:.0f}MB  "
            f"Batch Wait: {avg_wait_time:.1f}ms  "
            f"CPU: {avg_cpu_util:.2f}%  "
        )

        if train_metrics["gpu_utils"]:
            log_msg += f"GPU: {avg_gpu_util:.2f}%  "

        logging.info(log_msg)

        # Checkpoint and early stopping
        if val_metrics["eval_acc"] > best_accuracy or val_metrics['eval_loss'] < best_loss:
            if val_metrics["eval_acc"] > best_accuracy:
                best_accuracy = val_metrics["eval_acc"]
            if val_metrics['eval_loss'] < best_loss:
                best_loss = val_metrics['eval_loss']

            degradation = 0
            torch.save(model.state_dict(), f"{filepath}.pth")
            train_metrics["checkpoints"].append(epoch+1)
            logging.info("Validation metrics improved. Model saved.")
        else:
            degradation += 1
            logging.info(f"Validation metrics degraded for {degradation} epochs.")

            if degradation >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                train_metrics["early_stop"] = epoch+1
                break

    
    # Calculate overall averages
    avg_train_acc = sum(train_metrics["train_acc"]) / len(train_metrics["train_acc"]) if train_metrics["train_acc"] else 0
    avg_train_loss = sum(train_metrics["train_loss"]) / len(train_metrics["train_loss"]) if train_metrics["train_loss"] else 0
    avg_train_f1 = sum(train_metrics["train_f1"]) / len(train_metrics["train_f1"]) if train_metrics["train_f1"] else 0
    avg_val_acc = sum(train_metrics["val_acc"]) / len(train_metrics["val_acc"]) if train_metrics["val_acc"] else 0
    avg_val_loss = sum(train_metrics["val_loss"]) / len(train_metrics["val_loss"]) if train_metrics["val_loss"] else 0
    avg_val_f1 = sum(train_metrics["val_f1"]) / len(train_metrics["val_f1"]) if train_metrics["val_f1"] else 0
    avg_cpu_util = sum(train_metrics["cpu_utils"]) / len(train_metrics["cpu_utils"]) if train_metrics["cpu_utils"] else 0
    avg_gpu_util = sum(train_metrics["gpu_utils"]) / len(train_metrics["gpu_utils"]) if train_metrics["gpu_utils"] else 0
    avg_mem_util = sum(train_metrics["mem_utils"]) / len(train_metrics["mem_utils"]) if train_metrics["mem_utils"] else 0
    avg_wait_time = sum(train_metrics["batch_wait_times"]) / len(train_metrics["batch_wait_times"]) if train_metrics["batch_wait_times"] else 0

    log_msg = (
        "TRAINING RESULTS  "
        f"Avg Train Loss: {avg_train_loss:.4f}  "
        f"Avg Train Acc: {avg_train_acc:.2f}%  "
        f"Avg Train F1: {avg_train_f1:.4f}  "
        f"Avg Val Loss: {avg_val_loss:.4f}  "
        f"Avg Val Acc: {avg_val_acc:.2f}%  "
        f"Avg Val F1: {avg_val_f1:.4f}  "
        f"Avg Mem: {avg_mem_util:.0f}MB  "
        f"Avg Batch Wait: {avg_wait_time:.1f}ms  "
        f"Avg CPU: {avg_cpu_util:.2f}%  "
    )

    if train_metrics["gpu_utils"]:
        log_msg += f"Avg GPU: {avg_gpu_util:.2f}%  "

    logging.info(log_msg)

    logging.info(f"Best accuracy: {best_accuracy:.2f}%")

    return train_metrics
