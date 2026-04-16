import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if (not self.training) or torch.rand(1).item() > self.p:
            return x

        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        x_norm = (x - mu) / sigma

        perm = torch.randperm(b, device=x.device)
        mu2, sigma2 = mu[perm], sigma[perm]

        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        mu_mix = lam * mu + (1 - lam) * mu2
        sigma_mix = lam * sigma + (1 - lam) * sigma2
        return x_norm * sigma_mix + mu_mix


class BaselineCNN(nn.Module):
    # width_mult provides an explicit efficiency/performance tradeoff control.
    def __init__(self, width_mult=0.75, use_mixstyle=True):
        super().__init__()
        c1 = max(16, int(24 * width_mult))
        c2 = max(24, int(48 * width_mult))
        c3 = max(32, int(96 * width_mult))

        self.stage1 = nn.Sequential(DepthwiseSeparableConv(3, c1), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(DepthwiseSeparableConv(c1, c2), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(DepthwiseSeparableConv(c2, c3), nn.MaxPool2d(2))
        self.mixstyle = MixStyle(p=0.5) if use_mixstyle else nn.Identity()

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.mixstyle(x)
        x = self.stage3(x)
        return self.head(x)
