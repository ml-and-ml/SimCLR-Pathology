import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import torch


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}

        self.backbone = self.resnet_dict[base_model]
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)





#for future idea
class ParallelResnet(ResNet):
    def __init__(self):
        super(ParallelResnet, self).__init__(BasicBlock, [2,2,2,2])

    def push(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):
        out = []
        for c in range(len(x)):
            x_c = self.push(x[c])
            out.append(x_c.unsqueeze_(1))
        out = torch.cat(out, dim=1)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)

        return out





