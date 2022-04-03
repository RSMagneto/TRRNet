import torch
from torch.autograd import Variable
from torch import optim
from network import Encoder, Decoder
import ssim_loss
import dataloader

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    raise('CUDA is not available')

Encoder = Encoder(channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                  window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True)
Decoder = Decoder(channels=4, hidden_dim=32, layers=[4, 4], heads=[4, 4], head_dim=32,
                  window_size=8, downscaling_factors=[1, 1], relative_pos_embedding=True)

Encoder, Decoder = Encoder.to(device), Decoder.to(device)
torch.cuda.manual_seed(100)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
    elif classname.find('LayerNorm') != -1:
        torch.nn.init.constant_(m.weight.data, 1.0)
        torch.nn.init.constant_(m.bias.data, 0)

Encoder.apply(weights_init), Decoder.apply(weights_init)

optimizer_E = optim.Adam(Encoder.parameters(), lr=1e-4)
optimizer_D = optim.Adam(Decoder.parameters(), lr=1e-4)
scheduler_E = optim.lr_scheduler.StepLR(optimizer_E, step_size=100, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=100, gamma=0.5)
criterion1 = ssim_loss.SSIMLoss()
criterion2 = torch.nn.L1Loss()
torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False
train_loader = dataloader.train_loader
test_loader = dataloader.test_loader

def train(epoch):
    for i, train_data in enumerate(train_loader, 1):
        Encoder.train(), Decoder.train()
        Decoder.zero_grad()
        Encoder.zero_grad()
        LRMS, PAN, HRMS = train_data
        LRMS, PAN, HRMS = LRMS.to(device), PAN.to(device), HRMS.to(device)
        LRMS, PAN, HRMS = Variable(LRMS.to(torch.float32)/2047), Variable(PAN.to(torch.float32)/2047), Variable(HRMS.to(torch.float32)/2047)

        l_unique, l_common, p_unique, p_common = Encoder(LRMS, PAN)
        h = torch.cat([l_unique, (l_common+p_common)/2, p_unique], 1)
        output = Decoder(h)
        loss = 0.1 * criterion1(output, HRMS) + 0.9 * criterion2(output, HRMS) + 0.1 * criterion2(l_common, p_common)
        loss.backward(retain_graph=True)
        optimizer_E.step()
        optimizer_D.step()
        print('%d epoch %d iter, loss is %.6f' % (epoch+1, i, loss.item()))

@torch.no_grad()
def eval(sum_loss=0):
    for i, test_data in enumerate(test_loader, 1):
        Encoder.eval(), Decoder.eval()

        LRMS, PAN, HRMS = test_data
        LRMS, PAN, HRMS = LRMS.to(device), PAN.to(device), HRMS.to(device)
        LRMS, PAN, HRMS = Variable(LRMS.to(torch.float32)/2047), Variable(PAN.to(torch.float32)/2047), Variable(HRMS.to(torch.float32)/2047)

        l_unique, l_common, p_unique, p_common = Encoder(LRMS, PAN)
        h = torch.cat([l_unique, (l_common+p_common)/2, p_unique], 1)
        output = Decoder(h)

        loss = 0.1 * criterion1(output, HRMS) + 0.9 * criterion2(output, HRMS) + 0.1 * criterion2(l_common, p_common)
        sum_loss += loss.item()

    return sum_loss

best_loss = 10000

for epoch in range(300):
    train(epoch)
    scheduler_E.step()
    scheduler_D.step()
    if (epoch+1) % 25 == 0:
        torch.save(Encoder.state_dict(), 'output/model/{}e.pth'.format(epoch+1))
        torch.save(Decoder.state_dict(), 'output/model/{}d.pth'.format(epoch+1))
        sum_loss = eval()
        if sum_loss < best_loss:
            best_loss = sum_loss
            torch.save(Encoder.state_dict(), 'output/model/beste.pth')
            torch.save(Decoder.state_dict(), 'output/model/bestd.pth')
