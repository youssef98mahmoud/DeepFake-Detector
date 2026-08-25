#!/usr/bin/python3
'''Contains all data manipulation modules: download, sort, load.

   Requires a YAML file listing the dataset names (used as folder name), and
   kaggle URLs. For example:

   datasets:
    dataset1: "source/dataset1"
    dataset2: "source/dataset2"

   Kagglehub will automatically download datasets in the user home directory.
   This is not ideal when working in a team and/or on remote systems. By
   default, this script will use ./data_download, but you can also specify a
   location in the YAML file:

   kagglePath: "/new/location"
'''

import os
import shutil
import logging

import torch


def download(datasets):
    '''Downloads datasets listed in configuration from Kaggle.

    Args:
        datasets (dict): Datasets and Kaggle links to download.
    '''
    import kagglehub
    logging.info("Downloading data")

    for dataset_path in datasets.values():
        kagglehub.dataset_download(dataset_path)


def sort(source_path, dest_path, datasets):
    '''Merge downloaded datasets in one data folder and sort in fake/real
       directories:

       data/
         fake/
         real/

    Args:
        source_path (str): Kaggle download location.
        dest_path (str): Location of merged output dataset.
        datasets (dict): Datasets and Kaggle links to download.
    '''
    logging.info("Sorting data into fake and real directories at %s.", dest_path)

    if os.path.exists(dest_path):
        shutil.rmtree(dest_path)

    os.makedirs(dest_path)
    os.makedirs(os.path.join(dest_path, "fake"))
    os.makedirs(os.path.join(dest_path, "real"))

    for dataset_path in datasets.values():
        dataset_path = os.path.join(source_path, dataset_path)

        for root, _, files in os.walk(dataset_path):
            if files and files[0].endswith(".jpg"):
                img_dir = os.path.basename(root).lower()
                dest_dir = ""

                if any(txt in img_dir.lower() for txt in ["fake", "1"]):
                    dest_dir = os.path.join(dest_path, "fake")

                if any(txt in img_dir.lower() for txt in ["real", "0"]):
                    dest_dir = os.path.join(dest_path, "real")

                shutil.copytree(root, dest_dir, dirs_exist_ok=True)


def download_and_sort(config, dataset_path, downloading=False, sorting=False):
    '''Download data unless offline is specified.
       Sort data after download or if specified alongside offline.

    Args:
        config (_type_): _description_
        train_path (_type_): _description_
        test_path (_type_): _description_
    '''
    kaggle_path = os.path.join(os.environ["KAGGLEHUB_CACHE"], "datasets")

    if downloading:
        download(config)
        sort(kaggle_path, dataset_path, config)

    if sorting:
        sort(kaggle_path, dataset_path, config)


def load(data_src, img_height, img_width, augment=True):
    '''Loads, standardises, resizes, centeres, normalises and reformatts data
       into a Torch tensor. This dataset is then split into train and test
       datasets and split into batch sizes for training.

    Args:
        data_src (str): Path to image data.
        img_height (int): Spatial height size of training images.
        img_width (int): Spatial width size of training images.

    Returns:
        train_dataset: Torch dataset object of images.
        valid_dataset: Torch dataset object of images.
    '''
    import torchvision as torchvis

    if augment:
        return torchvis.datasets.ImageFolder(
            root=data_src,
            transform=torchvis.transforms.Compose([
                torchvis.transforms.Resize((img_height, img_width)),
                torchvis.transforms.RandomHorizontalFlip(p=0.5),
                torchvis.transforms.ColorJitter(brightness=0.2, contrast=0.2),
                #torchvis.transforms.RandomVerticalFlip(p=0.5),
                #torchvis.transforms.RandomRotation(10),
                #torchvis.transforms.CenterCrop((img_height, img_width)),
                torchvis.transforms.ToTensor(),
                torchvis.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )
    else:
        return torchvis.datasets.ImageFolder(
            root=data_src,
            transform=torchvis.transforms.Compose([
                torchvis.transforms.Resize((img_height, img_width)),
                torchvis.transforms.ToTensor(),
                torchvis.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        )


def split(dataset, split, seed):
    """
    Split dataset into train/val/test according to split dict:
    split = {"train": 0.7, "val": 0.2, "test": 0.1}
    """
    assert abs(sum(split.values()) - 1.0) < 1e-6, "Splits must sum to 1.0"

    # If dataset size is passed, split indices, otherwise split dataset obj
    if isinstance(dataset, int):
        train_size = int(dataset * split["train"])
        val_size = int(dataset * split["val"])
        test_size = dataset - train_size - val_size
        indices = list(range(dataset))

        generator = torch.Generator().manual_seed(seed)
        train_indices, val_indices, test_indices = torch.utils.data.random_split(
            indices, [train_size, val_size, test_size], generator=generator
        )

        return list(train_indices), list(val_indices), list(test_indices)
    else:
        train_size = int(len(dataset) * split["train"])
        val_size = int(len(dataset) * split["val"])
        test_size = len(dataset) - train_size - val_size

        return torch.utils.data.random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(seed)
        )


def process_img(image, img_process_funct):
    import numpy as np
    img = np.transpose(np.array(image[0]), (1, 2, 0))
    processed_img = img_process_funct(img)
    processed_img = np.transpose(processed_img, (2, 0, 1))

    post = (
        torch.tensor(processed_img, dtype=torch.float32),
        torch.tensor(image[1], dtype=torch.long)
    )

    return post


def preprocess(data, img_process_funct, cpus):
    import multiprocessing
    logging.info("Preprocessing data")
    logging.info("Total images to process: %s", len(data))

    proc_data = []

    with multiprocessing.Pool(processes=cpus) as pool:
        logging.info(f"Creating {cpus} threads for preprocessing images...")
        async_results = [
            pool.apply_async(process_img, [data[i], img_process_funct])
            for i in range(len(data))
        ]

        for result in async_results:
            try:
                processed_img = result.get()
            except Exception:
                logging.exception("A child process raised an exception")
            else:
                proc_data.append(processed_img)

    return proc_data


def create_dataloader(dataset, batch_size, number_workers):
    '''Split loaded into batch sizes for training.

    Args:
        dataset (obj): Torch dataset object of images.
        batch_size (int): Dataset batch size.
        number_workers (int): Number of dataloader workers.
        validation (bool): If False, dataset type is "Train", else "Valid".
                           Defaults to False.

    Returns:
        train_dataloader: Torch dataloader object of batched data for training.
        test_dataloader: Torch dataloader object of batched data for testing.
    '''
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=number_workers,
        pin_memory=True,
        persistent_workers=True if number_workers > 0 else False,
        prefetch_factor=4 if number_workers > 0 else None,
        #in_order=False if number_workers > 0 else True
    )

    return dataloader


def prepare_datasets(config, mode, data_path, seed, model):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    datasets = {}

    if "train" in mode:
        dataset_aug = load(train_path, config["imageHeight"], config["imageWidth"], augment=True)
        dataset_no_aug = load(train_path, config["imageHeight"], config["imageWidth"], augment=False)

        train_ds, val_ds, test_ds = split(len(dataset_aug), config["split"], seed)

        datasets["train"] = torch.utils.data.Subset(dataset_aug, train_ds)
        datasets["validate"] = torch.utils.data.Subset(dataset_no_aug, val_ds)
        datasets["test"] = torch.utils.data.Subset(dataset_no_aug, test_ds)

        if config["split"]["test"] == 0.0 and config.get("testData", None):
            datasets["test"] = load(test_path, config["imageHeight"], config["imageWidth"], augment=False)

        logging.info(f'Dataset split: train={len(datasets["train"])}, val={len(datasets["validate"])}, test={len(datasets["test"])} images')

    if mode == "infer":
        datasets["test"] = load(test_path, config["imageHeight"], config["imageWidth"], augment=False)

    for name, dataset in datasets.items():
        if config["modelType"] == "Quantum":
            dataset = preprocess(dataset, model.quanv_4, config["nCPUs"])

        datasets[name] = create_dataloader(dataset, config["batchSize"], config["numWorkers"])

    return datasets

