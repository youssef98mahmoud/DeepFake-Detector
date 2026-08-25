#!/usr/bin/python3
'''Load data and train model with parameters as specified in the configuration
   file.
'''
import os
import argparse
import logging
import shutil

import torch

from deepfake_detection import utils, data, train, eval, factory, visualise
from deepfake_detection.devices import classic_device


def argparser():
    '''Parses CLI arguments.

    Returns:
       ArgparseNamespace: Dictionary of CLI arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True,
                        help="Path to config YAML file.")
    parser.add_argument("--mode", "-m", type=str, default="train_infer",
                        help="Execution mode.",
                        choices=["train", "infer", "train_infer"])
    parser.add_argument("--download", "-d", action="store_true",
                        help=("Downloads and sorts data."))
    parser.add_argument("--sort", "-s", action="store_true",
                        help=("Use if changing split on pre-downloaded data "
                              "to avoid re-downloading."))
    parser.add_argument("--backbone", "-b", type=str, default=None,
                        help=("Path to off-the-shelf model weights."))
    parser.add_argument("--weights", "-w", type=str, default=None,
                        help=("Path to pre-trained model weights."))
    parser.add_argument("--dataPath", type=str, default="./data",
                        help=("Local path where datasets are stored."))
    parser.add_argument("--outputPath", type=str, default="./outputs",
                        help=("Directory for logs, checkpoints, and outputs."))
    parser.add_argument("--kagglePath", type=str, default="/tmp",
                        help=("Directory for Kaggle downloads."))

    return parser.parse_args()


def setup(config_path, kaggle_path, outputs_path):
    '''Sets up:
    - KAGGLE_CACHE path
    - Config dictionary, including checks
    - Session-specific outputs directory with a copy of the config
    - Logging
    - Seed

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    '''
    os.environ["KAGGLEHUB_CACHE"] = kaggle_path

    config = utils.check_config(utils.load_yaml(config_path))

    # Set up session-specific outputs directory
    datasets = "_".join(config["trainData"].keys())

    session_name = f"{config['modelType']}_bs{config['batchSize']}_ep{config['numEpochs']}_{datasets}_{args.mode}"

    if torch.cuda.is_available() and config["nGPUs"] > 0:
        session_name += f"_gpu{config['nGPUs']}"

    outputs_path = os.path.join(outputs_path, session_name)
    utils.check_path(outputs_path)

    # Make a copy of the configuration
    try:
        shutil.copyfile(config_path, os.path.join(outputs_path, f"{session_name}.conf.yaml"))
    except shutil.SameFileError:
        pass

    # Configure logging
    utils.setup_log(outputs_path, session_name, "info")
    logging.info("Loaded configuration: %s", config)

    # Set seed for reproducibility
    seed = config["manualSeed"]
    torch.manual_seed(seed)
    logging.info("Seed: %s", seed)

    return config, seed, outputs_path, session_name


def main(args):
    '''Main function loading config, loading data, splitting it into training
       and valing datasets, training and saving a model.

    Args:
        args (ArgparseNamespace): Dictionary of CLI arguments.
    '''
    # CLI arg checks
    if args.mode == "infer" and args.weights is None:
        input("Are you sure you want to run inference on the off-the-shelf model? If yes, press Enter to continue. If no, press Ctrl+C to stop and specify --weights.")

    # Environment setup
    config, seed, outputs_path, session_name = setup(args.config, args.kagglePath, args.outputPath)

    # Create device
    device_obj = classic_device.ClassicDevice(config["nGPUs"])
    device = device_obj.device
    num_gpus = device_obj.num_gpus

    logging.info(f"Device type: {device.type}")
    if device.type == "cuda":
        logging.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logging.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Create model
    model = factory.create_model(
        config,
        backbone_path=args.backbone,
        weights_path=args.weights,
        device=device
    )

    model = model.to(device)

    ## Parallelise data processing
    if num_gpus > 1:
        logging.info(f"Enabling DataParallel across {num_gpus} GPUs")
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
    elif num_gpus == 1:
        logging.info("Using single GPU")
    else:
        logging.info("Using CPU")

    ## Download and sort data
    if config.get("trainData", None):
        train_path = os.path.join(args.dataPath, "train")
        data.download_and_sort(config["trainData"], train_path, args.download, args.sort)

    if config.get("testData", None):
        test_path = os.path.join(args.dataPath, "test")
        data.download_and_sort(config["testData"], test_path, args.download, args.sort)

    # Prepare datasets
    datasets = data.prepare_datasets(config, args.mode, args.dataPath, seed, model)

    # mode in ["infer", "train", "train_infer"]
    if "train" in args.mode:
        logging.info("--STARTING TRAINING--")
        train_metrics = train.train(
            device, model, datasets["train"], datasets["validate"],
            config["numEpochs"], config["learningRate"], config["patience"],
            os.path.join(outputs_path, session_name), config["warmupEpochs"]
        )

        # Save CSV
        utils.save_results(
            train_metrics,
            f"{os.path.join(outputs_path, session_name)}.csv"
        )

        # Save visualization
        visualise.detailed_plot(
            train_metrics,
            f"{os.path.join(outputs_path, session_name)}_metrics.png"
        )

    if "infer" in args.mode:
        logging.info("--STARTING INFERENCE--")
        # Access underlying model if wrapped in DataParallel
        model = model.module if hasattr(model, 'module') else model
        # Add loss function to collect inference loss
        loss_fn = torch.nn.CrossEntropyLoss()
        eval_metrics = eval.eval_model(model, device, datasets["test"], loss_fn)

        # Log results
        avg_mem_util = sum(eval_metrics["mem_utils"]) / len(eval_metrics["mem_utils"]) if eval_metrics["mem_utils"] else 0
        avg_cpu_util = sum(eval_metrics["cpu_utils"]) / len(eval_metrics["cpu_utils"]) if eval_metrics["cpu_utils"] else 0
        avg_gpu_util = sum(eval_metrics["gpu_utils"]) / len(eval_metrics["gpu_utils"]) if eval_metrics["gpu_utils"] else 0
        avg_wait_time = sum(eval_metrics["batch_wait_times"]) / len(eval_metrics["batch_wait_times"]) if eval_metrics["batch_wait_times"] else 0

        log_msg = (
            "INFERENCE RESULTS  "
            f"Avg Eval Loss: {eval_metrics['eval_loss']:.4f}  "
            f"Avg Eval Acc: {eval_metrics['eval_acc']:.2f}%  "
            f"Avg Eval F1: {eval_metrics['eval_f1']:.4f}  "
            f"Avg Mem: {avg_mem_util:.0f}MB  "
            f"Avg Batch Wait: {avg_wait_time:.1f}ms  "
            f"Avg CPU: {avg_cpu_util:.2f}%  "
        )

        if eval_metrics["gpu_utils"]:
            log_msg += f"Avg GPU: {avg_gpu_util:.2f}%  "

        logging.info(log_msg)

        # Save visualization
        visualise.show_predictions(
            model, datasets["test"], device,
            f"{os.path.join(outputs_path, session_name)}_predictions.png"
        )


if __name__ == '__main__':
    args = argparser()
    main(args)
