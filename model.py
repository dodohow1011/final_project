import torch
import torch.nn as nn
import torchvision.models as models

class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()

        self.encoder1 = models.alexnet(pretrained=True)
        self.encoder2 = models.vgg16(pretrained=True)
        # self.encoder2 = models.resnext50_32x4d(pretrained=True)
        self.clf= nn.Sequential(
                            nn.Linear(2000, 256),
                            nn.ReLU(),
                            nn.Linear(256, 64),
                            nn.ReLU(),
                            nn.Linear(64, 16),
                            nn.ReLU(),
                            nn.Linear(16, 3),
                            nn.Softmax(dim=-1)
                          )

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        x3 = torch.cat((x1, x2), dim=-1)
        # x3 = 0.5 * (x1 + x2)
        x = self.clf(x3)

        return x, x3

class classifier_sep(nn.Module):
    def __init__(self):
        super(classifier_sep, self).__init__()

        self.encoder1 = models.alexnet(pretrained=True)
        self.encoder2 = models.vgg16(pretrained=True)
        # self.encoder2 = models.resnext50_32x4d(pretrained=True)
        self.clfs = nn.ModuleList([nn.Linear(2000, 1) for i in range(3)])

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x)
        x3 = torch.cat((x1, x2), dim=-1)

        out = torch.FloatTensor([]).cuda()
        for i in range(3):
            m = self.clfs[i](x3)
            out = torch.cat((out, m), dim=-1)

        return out, x3

