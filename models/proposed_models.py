import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class ProposedModel(nn.Module):

    def __init__(self):

        super().__init__()

        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # remove last classification layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # frequency branch
        self.freq = nn.Sequential(

            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )

        self.fc = nn.Sequential(

            nn.Linear(512 + 32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x,f):

        spatial = self.feature_extractor(x)

        spatial = torch.flatten(spatial,1)

        freq = self.freq(f)

        combined = torch.cat((spatial,freq),1)

        output = self.fc(combined)

        return output
