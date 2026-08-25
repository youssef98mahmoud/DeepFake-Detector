import torch
import torchvision as torchvis


class HQNN(torch.nn.Module):
    def __init__(self, device, backbone_path=None):
        super().__init__()

        # Classical model and layers
        if backbone_path:
            self.pretrainedmodel = torchvis.models.resnet18()
            self.pretrainedmodel.load_state_dict(torch.load(backbone_path))
            self.pretrainedmodel.eval()
        else:
            self.pretrainedmodel = torchvis.models.resnet18(
                weights=torchvis.models.ResNet18_Weights.DEFAULT
            )

        self.fc = torch.nn.Linear(1000, 417)
        self.fc1 = torch.nn.Linear(417, device.num_qubits * device.num_layers)

        self.quan_layers = torch.nn.ModuleList([
            device.create_layer()
            for _ in range(device.num_layers)
        ])

        self.fc2 = torch.nn.Linear(device.num_qubits * device.num_layers, 2)

    def forward(self, x):
        x = self.pretrainedmodel(x)

        x = torch.nn.functional.relu(self.fc(x))
        x = torch.nn.functional.relu(self.fc1(x))

        chunks = torch.split(x, self.fc1.out_features // len(self.quan_layers), dim=1)
        outputs = [layer(chunk) for layer, chunk in zip(self.quan_layers, chunks)]
        x = torch.cat(outputs, dim=1)
        x = torch.nn.functional.softmax(self.fc2(x), dim=1)

        return x
