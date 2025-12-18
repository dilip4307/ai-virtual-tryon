import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# FIXED & CONSISTENT GMM MODULE
# -----------------------------
# This version fixes the channel mismatch between
# correlation map and FeatureRegression.

class FeatureL2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        return x / (norm + 1e-6)


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc=3, ngf=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class FeatureRegression(nn.Module):
    # corr_map is single-channel
    def __init__(self, input_nc=1, output_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(128 * 6 * 4, output_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class GMM(nn.Module):
    def __init__(self, input_nc=3, output_dim=6):
        super().__init__()
        self.extractionA = FeatureExtraction(input_nc)
        self.extractionB = FeatureExtraction(input_nc)
        self.l2norm = FeatureL2Norm()
        self.regression = FeatureRegression(1, output_dim)

    def forward(self, person, cloth):
        featA = self.l2norm(self.extractionA(person))
        featB = self.l2norm(self.extractionB(cloth))

        B, C, H, W = featA.shape

        # Proper correlation map
        corr = torch.bmm(
            featA.view(B, C, H * W).transpose(1, 2),
            featB.view(B, C, H * W)
        )

        # Mean correlation → single-channel map
        corr_map = corr.mean(dim=2).view(B, 1, H, W)

        theta = self.regression(corr_map).view(-1, 2, 3)
        grid = F.affine_grid(theta, cloth.size(), align_corners=False)
        return grid, theta
