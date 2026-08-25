import numpy as np
import torch
import pennylane as qml


class QNN(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.qnode = qml.QNode(device.random_circuit, device.device)

        self.conv1 = torch.nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = torch.nn.MaxPool2d(2, 2)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, 2)

        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = torch.nn.MaxPool2d(2, 2)

        self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv9 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = torch.nn.MaxPool2d(2, 2)

        self.flatten = torch.nn.Flatten()

        self.fc1 = torch.nn.Linear(2048, 128)  # Adjust depending on input size
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.pool1(x)

        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = self.pool2(x)

        x = torch.nn.functional.relu(self.conv5(x))
        x = torch.nn.functional.relu(self.conv6(x))
        x = self.pool3(x)

        x = torch.nn.functional.relu(self.conv7(x))
        x = torch.nn.functional.relu(self.conv8(x))
        x = torch.nn.functional.relu(self.conv9(x))
        x = self.pool4(x)

        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)

        return x

    def quanv_4(self, image):
        height, width, _ = image.shape
        out = np.zeros((height // 2, width // 2, 4), dtype=np.float32)

        for j in range(0, height - 1, 2):
            for k in range(0, width - 1, 2):
                for l in range(_):
                    q_results = self.qnode(
                                    [
                                        image[j, k, l],
                                        image[j, k + 1, l],
                                        image[j + 1, k, l],
                                        image[j + 1, k + 1, l]
                                    ]
                                )
                    for c in range(4):
                        out[j // 2, k // 2, c] = q_results[c].item()

        return out
