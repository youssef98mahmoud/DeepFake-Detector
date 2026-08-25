import torch
import torchvision as torchvis


class ClassicalModel(torch.nn.Module):
    def __init__(self, backbone_path=None):
        super().__init__()

        if backbone_path:
            self.pretrainedmodel = torchvis.models.resnet18()
            self.pretrainedmodel.load_state_dict(torch.load(backbone_path))
            self.pretrainedmodel.eval()
        else:
            self.pretrainedmodel = torchvis.models.resnet18(
                weights=torchvis.models.ResNet18_Weights.DEFAULT
            )

        # Remove the final classification layer to use feature extractor
        self.pretrainedmodel.fc = torch.nn.Identity()

        # Custom classifier head
        self.fc = torch.nn.Linear(512, 64)
        self.fc2 = torch.nn.Linear(64, 2)

    def forward(self, x):
        x = self.pretrainedmodel(x)
        x = torch.nn.functional.relu(self.fc(x))
        x = self.fc2(x)

        return x
