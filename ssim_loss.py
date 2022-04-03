import torch
import torch.nn.functional as F
from math import exp

class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def create_window(self, window_size=11, sigma=1.5, channel=None):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, X, Y, window=None, group=None, scale=None):
        muX, muY = F.conv2d(X, window, groups=group), F.conv2d(Y, window, groups=group)
        muX_sq, muY_sq, muXY = muX * muX, muY * muY, muX * muY
        sigmaX_sq, sigmaY_sq, sigmaXY = F.conv2d(X * X, window, groups=group) - muX_sq, F.conv2d(Y * Y, window, groups=group) - muY_sq, F.conv2d(
            X * Y, window, groups=group) - muXY
        C1, C2 = (0.01 * scale) ** 2, (0.03 * scale) ** 2
        v1, v2 = 2.0 * sigmaXY + C2, sigmaX_sq + sigmaY_sq + C2
        ssim_map = ((2 * muXY + C1) * v1) / ((muX_sq + muY_sq + C1) * v2)
        result = ssim_map.mean()
        return result

    def forward(self, X, Y):
        window = self.create_window(channel=4).to(device=X.device, dtype=X.dtype)
        loss = 1 - self.ssim(X, Y, window, group=4, scale=1)
        return loss