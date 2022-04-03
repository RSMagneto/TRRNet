import torch
from swin_transformer import *

class Encoder(nn.Module):
    def __init__(self, *, channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                 window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True):
        super().__init__()

        self.Encoder = nn.Sequential(
            #CC
            nn.Conv2d(in_channels=channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*2, out_channels=hidden_dim*2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            #Downsample
            nn.AvgPool2d(2, 2),
            #TT
            StageModule(in_channels=hidden_dim*2, hidden_dimension=hidden_dim*2, layers=layers[1],
                        downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            StageModule(in_channels=hidden_dim*2, hidden_dimension=hidden_dim*2, layers=layers[1],
                        downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        )

    def forward(self, LRMS, PAN):
        l = self.Encoder(nn.functional.interpolate(LRMS, scale_factor=4))
        p = self.Encoder(torch.repeat_interleave(PAN, dim=1, repeats=4))
        l_unique, l_common = torch.chunk(l, 2, 1)
        p_unique, p_common = torch.chunk(p, 2, 1)
        return l_unique, l_common, p_unique, p_common

class Decoder(nn.Module):
    def __init__(self, *, channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                 window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True):
        super().__init__()

        self.Decoder = nn.Sequential(
            #TT
            StageModule(in_channels=hidden_dim*3, hidden_dimension=hidden_dim*3, layers=layers[0],
                        downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            StageModule(in_channels=hidden_dim*3, hidden_dimension=hidden_dim*3, layers=layers[0],
                        downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                        window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            #Upsample
            nn.Upsample(scale_factor=2),
            #CC
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=hidden_dim*3, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            #RefineLayer
            nn.Conv2d(in_channels=hidden_dim*3, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h):
        HRMS = self.Decoder(h)
        return HRMS