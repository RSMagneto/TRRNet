import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io

class RSI_Dataset(Dataset):
    def __init__(self, txtpath):
        super(RSI_Dataset, self).__init__()
        fp = open(txtpath, 'r')
        img_list = []
        for index in fp:
            index = index.rstrip('\n')
            img_list.append(index)
        self.img_list = img_list

    def __getitem__(self, index):
        fn = self.img_list[index]
        LRMS = io.imread('dataset/train/SimulatedRawMS/' + fn + 'ms.tif')
        PAN = io.imread('dataset/train/SimulatedRawPAN/' + fn + 'p.tif')
        HRMS = io.imread('dataset/train/ReferenceRawMS/' + fn + 'MSref.tif')
        LRMS = torch.FloatTensor(LRMS.astype(float)).permute(2, 0, 1)
        PAN = torch.FloatTensor(PAN.astype(float)).unsqueeze(0)
        HRMS = torch.FloatTensor(HRMS.astype(float)).permute(2, 0, 1)
        return LRMS, PAN, HRMS

    def __len__(self):
        return len(self.img_list)

train_data = RSI_Dataset('dataset/train.txt')
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=4)

test_data = RSI_Dataset('dataset/test.txt')
test_loader = DataLoader(dataset=test_data, batch_size=256, shuffle=False, num_workers=1)