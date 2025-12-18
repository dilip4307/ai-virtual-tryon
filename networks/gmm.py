import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / (norm + 1e-6)
        return x

class FeatureExtraction(nn.Module):
    def __init__(self, input_nc=3, ngf=64):
        super(FeatureExtraction, self).__init__()
        model = [
            nn.Conv2d(input_nc, ngf, 7, 1, 3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf*2, 4, 2, 1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2, ngf*4, 4, 2, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*4, ngf*4, 3, 1, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*4, ngf*4, 3, 1, 1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=6):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.linear = nn.Linear(256*6*4, output_dim)

    def forward(self, x):
        x = self.conv(x)
        # adaptive pool to match linear layer input
        x = F.adaptive_avg_pool2d(x, (6, 4))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class GMM(nn.Module):
    def __init__(self, input_nc=3, output_dim=6):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(input_nc)
        self.extractionB = FeatureExtraction(input_nc)
        self.l2norm = FeatureL2Norm()
        self.regression = FeatureRegression(512, output_dim)

    def forward(self, person, cloth):
        featA = self.extractionA(person)
        featB = self.extractionB(cloth)
        featA = self.l2norm(featA)
        featB = self.l2norm(featB)
        
        # correlation
        correlation = torch.bmm(
            featA.view(featA.size(0), featA.size(1), -1).transpose(1, 2),
            featB.view(featB.size(0), featB.size(1), -1)
        )
        correlation = correlation.view(
            featA.size(0), featA.size(2), featA.size(3),
            featB.size(2), featB.size(3)
        )
        corr_map = torch.mean(correlation, dim=(3, 4))  # [B, C, H, W]

        theta = self.regression(corr_map)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, cloth.size(), align_corners=False)
        return grid, theta
