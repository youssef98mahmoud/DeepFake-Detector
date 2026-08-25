import torch

class ClassicDevice:
    '''Create and return torch device.

    Returns:
        device (obj): torch device
    '''
    def __init__(self, num_gpus):
        if torch.cuda.is_available() and torch.cuda.device_count() >= num_gpus:
            self.device = torch.device("cuda")
            self.num_gpus = num_gpus
        else:
            self.device = torch.device("cpu")
            self.num_gpus = 0

