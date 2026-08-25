# deepfake-detection

## ToC

- [deepfake-detection](#deepfake-detection)
  - [ToC](#toc)
  - [Background](#background)
    - [Objective](#objective)
    - [Metrics](#metrics)
    - [Research](#research)
  - [Model choice](#model-choice)
  - [Data choice](#data-choice)
  - [Device choice](#device-choice)
  - [Environment choice](#environment-choice)
    - [Local](#local)
    - [Docker](#docker)
    - [GPU services](#gpu-services)
  - [Configuration](#configuration)
    - [Examples](#examples)
  - [Usage](#usage)
    - [CLI arguments](#cli-arguments)
    - [Examples](#examples-1)
  - [Outputs](#outputs)
    - [Interpreting outputs](#interpreting-outputs)
      - [Overview](#overview)
      - [Loss per Epoch](#loss-per-epoch)
    - [Accuracy per Epoch](#accuracy-per-epoch)

## Background

### Objective

The main aim of the project is to create a classifier capable of detecting whether an image is real or fake (i.e., AI generated) while showing the difference in performance between classic vs quantum Machine Learning.  

The approach requires the creation and benchmarking of a classic classification model capable of detecting deepfakes followed by the addition of quantum layers to create a hybrid model and its benchmarking for comparison.

### Metrics

When benchmarking the models, a set of metrics is selected to ensure fair comparison:

* Prediction accuracy
* Training loss

For scalability benchmarking, the following metrics will be considered:

* Type of optimiser
* Number of training epochs
* Training batch size
* Resource usage (e.g., GPU usage)

Specifically for quantum benchmarking:

* Number of qubits
* Number of gates
* Circuit depth

Optionally, if time allows, make use of explainable methods to collect information about the features used in the learning process for comparison.

### Research

* [Join the High Accuracy Club on ImageNet with A Binary Neural Network Ticket](https://arxiv.org/abs/2211.12933)
* [Faster Than Lies: Real-time Deepfake Detection using Binary Neural Networks](https://arxiv.org/abs/2406.04932)
* [CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images](https://arxiv.org/abs/2303.14126#)
* [Quantum-Trained Convolutional Neural Network for Deepfake Audio Detection](https://arxiv.org/html/2410.09250v1)
* [Subspace Preserving Quantum Convolutional Neural Network Architectures](https://arxiv.org/abs/2409.18918)
* [Hybrid quantum image classification and federated learning for hepatic steatosis diagnosis](https://arxiv.org/abs/2311.02402)
* [Quantum machine learning for image classification](https://arxiv.org/abs/2304.09224)
* [A Novel Quantum Neural Network Approach to Combating Fake Reviews](https://d1wqtxts1xzle7.cloudfront.net/114003957/s44227_024_00028_x-libre.pdf?1714538109=&response-content-disposition=inline%3B+filename%3DA_Novel_Quantum_Neural_Network_Approach.pdf&Expires=1730373564&Signature=GdYaC9wub7QV6P~81xb9AMdYeNl3Hsk1tqh4CoajaJHeJya~PNGdTNV00yeACXuJW1epx0REB73K0VlP8YnXUjBeBybBpl49N-F4zvTkwgwqGc0fMg7WK1JanuD6J6wIMdRI5RVy39GUVdwOPHjHE-tmWXeoQ1owc2NP1y3SHQoA-z1Y0wkWaL93uh~LB45vD9Fc~3ILkib9FGnMiIasuEYtY6A9OmZIxi00SLVlvEz31s1qj8FZgeeKZ5HZBo0seIVWtyofcnkDH-JF1nYckW9L3VTFP46ju8JZpgjiT07pD8uHWpVzR6SL8oucH2CUQmGyCo3bhcDMOwbIPwPb1A__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
* [Pooling techniques in hybrid quantum-classical convolutional neural networks](https://arxiv.org/pdf/2305.05603)

## Model choice

The problem requires a classification model, and due to the fact that we need to classify whether an image is real or fake, this fits very well with Binary Neural Networks.  

Implemented model types:

| Model | Type | Description |
| ----- | ---- | ----------- |
| Classic | Classic CNN | Uses a pretrained ResNet‑18 to extract visual features and a classifier to decide between two classes: real vs fake. |
| Hybrid | Hybrid Classic-Quantum NN | Uses ResNet‑18 for feature extraction, then sends the features through several quantum layers that mimic how qubits process information before making a prediction. |
| Quantum* | Quantum CNN | Processes images with a deep stack of standard convolutional layers, then applies a “quantum filter” that analyzes small patches using a quantum circuit. |
| Spiking | Spiking CNN + HQNN | Extracts features with ResNet‑18, then uses a spiking‑neuron controller to decide how a quantum‑inspired circuit should behave. The quantum circuit runs multiple times with feedback, and the combined output is classified. |

>\* The Quantum model has not been updated to match the latest changes in device design and thus is currently unoperable. The new Spiking model was prioritised due to the Quantum model taking too long to pre-process images and having the lowest accuracy levels in previous runs.

## Data choice

The focus will initially be on facial deepfake detection as there are many open-source datasets available for this use case. Ideally, we would like to expand this to include other types of images.

| Dataset | Purpose | Kaggle handle | Structure |
| ------- | ------- | ------------- | --------- |
| [1k](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces)  | testing | "hamzaboulahia/hardfakevsrealfaces" | `fake/`, `real/` |
| [2k](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection) | training | "ciplab/real-and-fake-face-detection" | `real_and_fake_face/training_fake/`, `real_and_fake_face/training_real/` |
| [10k*](https://www.kaggle.com/datasets/sachchitkunichetty/rvf10k) | training and testing | "sachchitkunichetty/rvf10k" | `rvf10k/train/fake/`, `rvf10k/train/real/`, `rvf10k/valid/fake/`, `rvf10k/valid/real/` |
| [120k](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | training and testing | "birdy654/cifake-real-and-ai-generated-synthetic-images" |  |
| [140k](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | training and testing | "xhlulu/140k-real-and-fake-faces" | `real_vs_fake/real_vs_fake/train/fake/`, `real_vs_fake/real_vs_fake/train/real/`, `real_vs_fake/real_vs_fake/valid/fake/`, `real_vs_fake/real_vs_fake/valid/real/`, `real_vs_fake/real_vs_fake/test/fake/`, `real_vs_fake/real_vs_fake/test/real/` |
| [162k](https://www.kaggle.com/datasets/wish096/realvsfake-81k-by-wish) | training and testing | "wish096/realvsfake-81k-by-wish" |  |

>\* `10k` is a subset of `140k`

>**Note:** Different directory structures pose a challenge for automatic download and loading of public datasets. We developed a way to merge two or more datasets and then programmatically split them into training, validation, and inference. While this is a good way to create a larger dataset, it poses a problem in that datasets such as `hard1k` are designed for inference only, so care must be taken when configuring the datasets.

## Device choice

The classic model can run on standard devices like CPU and GPU with parallel processing, but Quantum-based models require a Quantum simulator or device.

Implemented quantum devices:

| Device | Type |
| ------ | ---- |
| `default.qubit` | PennyLane CPU-based quantum simulator |
| `lightning.gpu` | PennyLane GPU-based quantum simulator |
| `qiskit.aer` | Qiskit CPU-based noisy quantum simulator |
| `qiskit.transpile` | Qiskit CPU-based transpiler |
| `qiskit.hardware` | Qiskit interface to IBM real quantum hardware |

>**Note:** `qiskit.transpile` and `qiskit.hardware` require an `IBM_QUANTUM_API_KEY` to be sourced in your environment or saved in the root of this repository as `.ibm_api.env`.

## Environment choice

### Local

To run the code locally, clone the repository and install the requirements in a virtual environment:

```console
cd deepfake-detection
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Docker

Build the image and tag it with the version:

```console
docker build -t deepfake:v<X> .
```

To test the image locally, run the container with the three expected mounts:

```console
docker run -v .:/safe_data/deepfake
           -v ./results:/safe_outputs
           -v ./scratch:/scratch
           deepfake:v<X>
```

>**Note:** To test in offline mode (e.g., to simulate HPC systems), add `--network none` to the docker run command.

You can make this image accessible to pull and use from any system by pushing it to a Container Registry. Check the [guide](./REGISTRY.md) on how to do this.

### GPU services

- [EIDF](./kueue/eidf/README.md)
- [SHS](./kueue/shs/README.md)
- [Cirrus](./slurm/README.md#cirrus)
- [Isambard](./slurm/README.md#isambard)

## Configuration

Both the notebook and script require a configuration file such as [Deepfake.conf.yaml](./Deepfake.conf.yaml), but you can define your own. Here are the recognised parameters:

| Parameter | Required | Description | Format | Default |
| --------- | -------- | ----------- | ------ | ------- |
| `modelType` | **Yes** | Type of model to train or test. | One of `Classic`, `Hybrid`, `Quantum`, `Spiking` |  |
| `trainData` | only if `testData` is not provided | Dataset(s) to be downloaded for training. | [dataset_name: "Kaggle handle"] |  |
| `testData` | only if `split` test ratio is `0.0` | Dataset(s) to be downloaded for testing. | [dataset_name: "Kaggle handle"] |  |
| `split` | No | train/val/test ratios | Dictionary with a float value for each subset. See examples. | `80/20/0` if `testData` provided, `70/20/10` otherwise |
| `manualSeed` | No | Used for training reproducibility. | int between 1 and 10,000 | `999` |
| `numWorkers` | No | Number of workers for dataloader(s). | int | `2` |
| `batchSize` | No | Batch size during training. | int | `32` |
| `imageHeight` | No | Spatial height size of training images. All images will be resized to this size using a transformer. | int | `32` |
| `imageWidth` | No | Spatial width size of training images. All images will be resized to this size using a transformer. | int | `32` |
| `numChannels` | No | Number of channels in the training images. For color images this is 3. | int - 2 for B&W, 3 for RGB | `3` |
| `numEpochs` | No | Number of training epochs. | int | `10` |
| `learningRate` | No | Learning rate for optimizers. | float | `0.0002` |
| `beta1` | No | Beta1 hyperparameter for Adam optimizers. | float | `0.5` |
| `nGPUs` | No | Number of GPUs to be used of the available. | int | `0` |
| `nCPUs` | No | Number of CPUs to be used for image preprocessing. | int | `4` |
| `patience` | No | Number of epochs training degradation is allowed for before early stopping. | int | `5` |
| `warmupEpoch` | No | Number of epochs to apply linear warmup for before cosine decay. Only applies if `numEpochs` > 1,000, else only cosine decay will be applied. | int | `5` |

Quantum specific parameters - will be ignored if modelType is `Classic`:

| Parameter | Required | Description | Format | Default |
| --------- | -------- | ----------- | ------ | ------- |
| `quantumDevice` | No | Type of Quantum device. | One of `default.qubit`, `lightning.gpu`, `qiskit.aer`, `qiskit.transpile`, `qiskit.hardware:<backend>` | `default.qubit` |
| `nQubits` | No | Number of Qubits. | int | `4` |
| `diffType` | No | Diff method for training. | One of `backprop`, `adjoint`, `parameter-shift`, `default`, `best` | `best` |
| `nLayers` | No | Number of layers (1 if random). | int | `6` |
| `typeLayers` | No | Type of layers. | One of `basic` or `strong`. | `strong` |
| `optimisationLevel` | No | Transpilation optimisation of circuit. | int between 0 and 3 | `1` |
| `shots` | No | How many times the circuit runs to estimate the measurement results. | int between 1024 and 8192 | `1024` |
| `patch` | No | Specific to `QNN` models. Defines whether to use patched method for trainable QNN. | bool | `False` |

If you want to keep a parameter's default value, you can omit it from your configuration.

### Examples

This configuration will run the Classic model training and inference on the `deepfake2k` dataset with 70/20/10 train/val/test split, `batchSize` 32, `imageSize` 32x32, 3 channels, over 10 epochs with `learningRate` 0.0002 and `patience` of 5, on 4 CPUs:

```yaml
modelType: "Classic"
trainData:
  deepfake2k: "ciplab/real-and-fake-face-detection"
```

Configuring and using a different dataset for inference:

```yaml
modelType: "Classic"
trainData:
  deepfake2k: "ciplab/real-and-fake-face-detection"
testData:
  hard1k: "hamzaboulahia/hardfakevsrealfaces"
numWorkers: 4
batchSize: 64
imageHeight: 64
imageWidth: 64
numEpochs: 50
nGPUs: 1
patience: 5
```

Note the train/val/test split is going to default to 80/20/0 because `testData` will be used for inference.

Configuring a different split:

```yaml
modelType: "Hybrid"
trainData:
  deepfake2k: "ciplab/real-and-fake-face-detection"
split:
  train: 0.8
  val: 0.1
  test: 0.1
numWorkers: 4
batchSize: 64
imageHeight: 64
imageWidth: 64
numEpochs: 50
nGPUs: 1
patience: 5
quantumDevice: "lightning.gpu"
nQubits: 8
```

For more examples, check the [outputs](./outputs) folder.

## Usage

You can run the code via [Jupyter Notebook](Deepfake.ipynb), or via the [Python script](deepfake_detection.py).

### CLI arguments

| Argument | Required | Description | Format | Default |
| -------- | -------- | ----------- | ------ | ------- |
| -c, --config | **Yes** | Path to config YAML file.| str | |
| -m, --mode | No | Execution mode. | One of `train`, `infer`, `train+infer`. | `train+infer` |
| -d, --download | No | Downloads and sorts data. | bool - if used then True | False |
| -s, --sort | No | Use if changing split on pre-downloaded data to avoid re-downloading. | bool - if used then True | False |
| -b, --backbone | No | Path to off-the-shelf model weights for running offline. | str | None |
| -w, --weights | No | Path to pre-trained model weights for running inference. | str | None |
| --dataPath | No | Local path where datasets are stored. | str | `./data` |
| --outputPath | No | Directory for logs, checkpoints, and outputs. | str | `./outputs` |
| --kagglePath | No | Directory for Kaggle downloads. | str | `/tmp` |

### Examples

To run training and inference:

```console
python deepfake_detection.py -c Deepfake.conf.yaml -d
```

Data download and sorting are expensive tasks that are not required to run every time you train or test a model, so they will only take place when you specify the `-d` or `--download` flag.

You may want to change the data split on the same downloaded dataset. To do this, use the `-s` or `--sort` flag:

```console
python deepfake_detection.py -c Deepfake.conf.yaml -s
```

To run inference-only on a pre-trained model, use the same configuration as used for training, but with `infer` mode:

```console
python deepfake_detection.py -d -m infer \
                             -c outputs/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml \
                             -w outputs/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.pth
```

Some systems' compute nodes don't have internet access, for this use the `-b` or `--backbone` flag to specify the location of a downloaded model.

```console
python deepfake_detection.py -m infer \
                             -c outputs/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.conf.yaml \
                             -b ~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
                             -w outputs/Classic_bs32_ep5_hard1k/Classic_bs32_ep5_hard1k.pth
```

## Outputs

The deepfake detection program will create a session-specific folder with all the outputs at the configured `outputsPath`. Each session will contain the following outputs:

* Copy of the configuration file
* Log
* Trained model parameters
* CSV file of average accuracy and loss for every epoch
* Plot of average accuracy, loss, and resource use
* Plot of inference predictions

For an example of this, check the [outputs](./outputs) folder.

>**Note:** A session's name is not unique, with format `<modelType>_<batchSize>_<numEpochs>_<trainData>_<mode>`. This is by design to reduce the amount of data pushed to the repository. However, it does mean that if you run again with a slightly different configuration your outputs will be overridden. This is why you should push outputs to the repository between runs to make sure you save outputs and differences between runs.

### Interpreting outputs

Here is an example metric plot for a run on EIDF with the following configuration:

```yaml
modelType: "Classic"
trainData:
  120k: "birdy654/cifake-real-and-ai-generated-synthetic-images"
  1k: "hamzaboulahia/hardfakevsrealfaces"
split:
  train: 0.7
  val: 0.2
  test: 0.1
numWorkers: 16
batchSize: 512
learningRate: 0.0002
imageHeight: 224
imageWidth: 224
numEpochs: 50
patience: 5
nGPUs: 2
```

![Classic_bs512_ep50_cifake_hard1k_train_infer_gpu2_a10040_pvc](./outputs/Classic_bs512_ep50_cifake_hard1k_train_infer_gpu2_a10040_pvc/Classic_bs512_ep50_cifake_hard1k_train_infer_gpu2_metrics.png)

#### Overview

Each panel tracks key metrics across training epochs, with checkpoint markers (⭐) indicating epochs where the model achieved best validation performance and was checkpointed.

Early stopping is also represented on each panel as a vertical red dotted line.

#### Loss per Epoch

This panel shows training and validation loss over time.

| Observation | Interpretation |
| - | - |
| Both curves decline | the model is learning patterns |
| Curves converge | loss plateaus when model has learned as much as possible from the data |
| Small gaps between train and val curves | healthy generalisation |
| Large gaps between train and val curves | overfitting |
| Val loss increasing while train loss decreasing | overfitting |
| Both losses plateau early at high values | underfitting or LR too low |
| Erratic validation loss | unstable training or too few validation samples |

For the example graph, the validation loss tracks the training loss closely, the checkpoint at epoch 4 shows early 

### Accuracy per Epoch

TBC