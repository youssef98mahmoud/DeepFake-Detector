#!/usr/bin/python3
'''Contains utility code such as loggers, file opening operations, etc.
'''
import os
import logging
import subprocess
import psutil
import random
import numpy as np
import torch

import yaml
import pandas as pd


def setup_log(log_path, log_name, level):
    '''Configures logging.

    Args:
        log_path (str): Location of log file.
        log_name (str): Log file prefix (will be followed by timestamp and PID).
        level (str): noset|debug|info|warning|error|critical
    '''
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    filename = os.path.join(f"{log_path}", (f'{log_name}.log'))

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper()),
        handlers=[
            logging.FileHandler(filename, mode="w"),
            logging.StreamHandler()
        ]
    )


def load_yaml(yaml_path):
    '''Import YAML file.

    Args:
        yaml_path (str): Path to YAML file.

    Returns:
        Any: YAML contents.
    '''
    with open(yaml_path, "r", encoding="utf8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    return yaml_data


def check_config(config):
    '''Validate configuration.

    Args:
        config (dict): Configuration as loaded from YAML.

    Raises:
        ValueError: If required vars are missing or invalid.

    Returns:
        config (dict): Validated configuration.
    '''
    required_vars = {
        "trainData": "dict of datasets and Kaggle download paths",
        "modelType": "Classic, Hybrid, or Quantum",
    }

    optional_vars = {
        "manualSeed": 999,
        "numWorkers": 2,
        "batchSize": 32,
        "imageHeight": 32,
        "imageWidth": 32,
        "numChannels": 3,
        "numEpochs": 10,
        "learningRate": 0.0002,
        "beta1": 0.5,
        "nGPUs": 0,
        "nCPUs": 4,
        "patience": 5,
        "quantumDevice": "default.qubit",
        "nQubits": 4,
        "diffType": "best",
        "nLayers": 6,
        "typeLayers": "strong",
        "patch": False,
        "optimisationLevel": 1,
        "shots": 1024,
        "warmupEpochs": 5,
    }

    # Check required vars
    for required_var, required_val in required_vars.items():
        if required_var not in config:
            raise ValueError(f"{required_var} must be specified in the configuration as {required_val}")

    # Fill optional vars
    for optional_var, default_val in optional_vars.items():
        if optional_var not in config:
            config[optional_var] = default_val

    # Conditional split defaults
    if not config.get("split", None):
        if config.get("testData", None):
            config["split"] = {"train": 0.8, "val": 0.2, "test": 0.0}
        else:
            config["split"] = {"train": 0.7, "val": 0.2, "test": 0.1}

    for key in ["train", "val", "test"]:
        if key not in config["split"]:
            raise ValueError(f"split must contain '{key}' ratio")

        if config["split"][key] < 0:
            raise ValueError(f"split['{key}'] must be >= 0")

    total = config["split"]["train"] + config["split"]["val"] + config["split"]["test"]
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, but got {total}")

    # If test split is zero, user must provide separate testData
    if config["split"]["test"] == 0 and "testData" not in config:
        raise ValueError("split['test'] is 0, so you must provide testData for inference.")

    # If test split > 0, user must NOT provide testData (avoid conflicts)
    if config["split"]["test"] > 0 and "testData" in config:
        raise ValueError("You provided both split['test'] > 0 and testData. Choose one test source.")

    # Validate modelType
    if config["modelType"] not in ["Classic", "Hybrid", "Quantum", "Spiking"]:
        raise ValueError("modelType must be one of: Classic, Hybrid, Quantum")

    # Validate optimisationLevel
    if config["optimisationLevel"] not in range(0, 4):
        raise ValueError("Choose a valid optimisationLevel: 0 - none, 1 - light (default), 2 - medium, 3 - heavy")

    return config


def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
        logging.info(f"No seed provided. Using random seed: {seed}")
    else:
        logging.info(f"Using provided seed: {seed}")

    random.seed(seed)                      # Python random
    np.random.seed(seed)                   # NumPy
    torch.manual_seed(seed)                # PyTorch CPU
    torch.cuda.manual_seed(seed)           # PyTorch GPU
    torch.cuda.manual_seed_all(seed)       # All GPUs

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_path(dir_path):
    '''Checks if directory exists at given path and creates it if not.

    Args:
        dir_path (str): Directory path
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_results(data, filepath):
    df = pd.DataFrame({
        "Epoch": range(1, len(data["train_loss"]) + 1),
        "Train Loss": data["train_loss"],
        "Train Accuracy": data["train_acc"],
        "Train F1": data["train_f1"],
        "Val Loss": data["val_loss"],
        "Val Accuracy": data["val_acc"],
        "Val F1": data["val_f1"],
        "Mem Util": " ".join(str(x) for x in data["mem_utils"]),
        "CPU Util": " ".join(str(x) for x in data["cpu_utils"]),
        "GPU Util": " ".join(str(x) for x in data["gpu_utils"]),
    })

    df.to_csv(filepath, index=False)


def get_system_util(device):
    utils = {
        "cpu": psutil.cpu_percent(interval=0.1),
        "gpu": None
    }

    # Only show util of allocated CPUs
    if "SLURM_CPUS_PER_TASK" in os.environ:
        allocated_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        total_cpus = psutil.cpu_count()
        utils["cpu"] = utils["cpu"] * (total_cpus / allocated_cpus)

    if device.type == "cuda":
        try:
            import pynvml
            import time

            # Initialize once
            if not hasattr(get_system_util, '_nvml_init'):
                pynvml.nvmlInit()
                get_system_util._nvml_init = True

            device_count = pynvml.nvmlDeviceGetCount()
            all_samples = []

            # Take 3 samples over time
            for _ in range(3):
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    all_samples.append(util.gpu)

                if _ < 2:
                    time.sleep(0.02)  # 20ms between samples

            if all_samples:
                utils["gpu"] = sum(all_samples) / len(all_samples)
        except Exception as e:
            # Fallback to nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0 and result.stdout.strip():
                    lines = result.stdout.strip().split('\n')
                    gpu_values = [float(line.strip()) for line in lines if line.strip()]
                    if gpu_values:
                        utils["gpu"] = sum(gpu_values) / len(gpu_values)
            except Exception:
                pass

    return utils


def get_memory_util(device):
    if device.type == "cuda":
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2
            )

            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                total_memory = sum(float(line.strip()) for line in lines if line.strip())

                return total_memory
        except Exception as e:
            logging.debug(f"nvidia-smi unavailable ({e}), falling back to PyTorch")

        # Fallback: Use PyTorch
        total_memory = 0
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.memory_allocated(i) / 1024**2

        return total_memory if total_memory > 0 else torch.cuda.memory_allocated() / 1024**2
    else:
        # CPU memory
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**2

