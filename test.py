import os
import torch
from torch.autograd import Variable
from network import *
import numpy
from skimage import io

if torch.cuda.is_available:
    device = 'cuda:0'
else:
    raise('CUDA is not available')

Encoder = Encoder(channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                  window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True)
Decoder = Decoder(channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                  window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True)

Encoder, Decoder = Encoder.to(device), Decoder.to(device)
Encoder.load_state_dict(torch.load('output/model/beste.pth'))
Decoder.load_state_dict(torch.load('output/model/bestd.pth'))

with torch.no_grad():
    for i in range(1, 51):
        LRMS = torch.tensor(io.imread('dataset/test/Real/' + str(i) + 'ms.tif').astype(float)).to(device).permute(2, 0, 1).unsqueeze(0)
        PAN = torch.tensor(io.imread('dataset/test/Real/' + str(i) + 'p.tif').astype(float)).to(device).unsqueeze(0).unsqueeze(0)
        LRMS = Variable(LRMS.to(torch.float32)/2047)
        PAN = Variable(PAN.to(torch.float32)/2047)

        l_unique, l_common, p_unique, p_common = Encoder(LRMS, PAN)
        h = torch.cat([l_unique, (l_common+p_common)/2, p_unique], 1)
        output = Decoder(h)

        img = (output[0].detach() * 2047).clamp(0, 2047).permute(1, 2, 0).cpu().numpy().astype(numpy.uint16)
        fname = os.path.join('output/image/Real/', '{}.tif'.format(i))
        io.imsave(fname, img)

with torch.no_grad():
    for i in range(1, 51):
        LRMS = torch.tensor(io.imread('dataset/test/Simulated/' + str(i) + 'ms.tif').astype(float)).to(device).permute(2, 0, 1).unsqueeze(0)
        PAN = torch.tensor(io.imread('dataset/test/Simulated/' + str(i) + 'p.tif').astype(float)).to(device).unsqueeze(0).unsqueeze(0)
        HRMS = torch.tensor(io.imread('dataset/test/Simulated/' + str(i) + 'MSref.tif').astype(float)).to(device).permute(2, 0, 1).unsqueeze(0)
        LRMS = Variable(LRMS.to(torch.float32)/2047)
        PAN = Variable(PAN.to(torch.float32)/2047)
        HRMS = Variable(HRMS.to(torch.float32)/2047)

        l_unique, l_common, p_unique, p_common = Encoder(LRMS, PAN)
        h = torch.cat([l_unique, (l_common+p_common)/2, p_unique], 1)
        output = Decoder(h)

        img = (output[0].detach() * 2047).clamp(0, 2047).permute(1, 2, 0).cpu().numpy().astype(numpy.uint16)
        fname = os.path.join('output/image/Simulated/', '{}.tif'.format(i))
        io.imsave(fname, img)