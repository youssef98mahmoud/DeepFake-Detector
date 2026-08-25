#!/usr/bin/python3
from deepfake_detection import utils

import torch
import time
import logging
from sklearn.metrics import f1_score


def eval_model(model, device, dataset, loss_fn=None):
    eval_metrics = {
        "eval_acc": 0,
        "eval_loss": 0,
        "eval_f1": 0,
        "mem_utils": [],
        "cpu_utils": [],
        "gpu_utils": [],
        "batch_wait_times": [],
    }

    correct = 0
    loss = 0.0
    total_samples = 0

    # For F1 calculation
    all_eval_preds = []
    all_eval_labels = []

    model.eval()
    model = model.to(device)

    # Track batch loading time
    iter_start = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataset):
            # Measure batch wait time
            batch_ready_time = time.time()
            wait_time = batch_ready_time - iter_start
            eval_metrics["batch_wait_times"].append(wait_time * 1000)

            if batch_idx == 0:
                logging.info(f"First eval batch wait: {wait_time*1000:.1f}ms")

            # Eval
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if loss_fn:
                loss += loss_fn(outputs, labels).item() * labels.size(0)

            preds = torch.argmax(outputs, axis=1)
            correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # Collect predictions and labels for F1 score
            all_eval_preds.extend(preds.cpu().numpy())
            all_eval_labels.extend(labels.cpu().numpy())

            # Collect resource util
            batch_utils = utils.get_system_util(device)
            eval_metrics["cpu_utils"].append(batch_utils["cpu"])

            if batch_utils["gpu"] is not None:
                eval_metrics["gpu_utils"].append(batch_utils["gpu"])

            eval_metrics["mem_utils"].append(utils.get_memory_util(device))

            # Start timer for next batch
            iter_start = time.time()

    # Calculate overall averages
    eval_metrics["eval_acc"] = 100 * correct / total_samples
    eval_metrics["eval_loss"] = loss / total_samples if loss_fn else None
    eval_metrics["eval_f1"] = f1_score(all_eval_labels, all_eval_preds, average='binary')

    return eval_metrics
