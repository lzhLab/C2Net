import torch
from torch import nn
import os,cv2

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class RNN_Model(nn.Module):
    scale = 32

    def __init__(self, in_ch, out_ch, **args):
        super(RNN_Model, self).__init__()
        self.conv1 = DoubleConv(in_ch, self.scale)
        self.conv2 = DoubleConv(self.scale, self.scale * 2) 
        self.conv3 = DoubleConv(self.scale * 2, self.scale * 4)
        self.conv4 = DoubleConv(self.scale * 4, self.scale * 8)

        self.conv5 = DoubleConv(self.scale * 8 + self.scale * 4, self.scale * 4)
        self.conv6 = DoubleConv(self.scale * 4 + self.scale * 2, self.scale * 2)
        self.conv7 = DoubleConv(self.scale * 2 + self.scale, self.scale)
        self.conv8 = nn.Conv2d(self.scale, out_ch, 1)

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1_1 = nn.Conv2d(self.scale * 8, 1, kernel_size=1, padding=0, bias=False)
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, liver_imgs, vessel_mask, att):
        merge2 = self.pool(att)

        c1 = self.conv1(liver_imgs)
        p1 = self.pool(c1)
        merge_c1 = p1 * merge2
        p1 = p1 + merge_c1
        
        c2 = self.conv2(p1)
        p2 = self.pool(c2)
        merge3 = self.pool(merge2)
        merge_c2 = p2 * merge3
        p2 = p2 + merge_c2
        
        c3 = self.conv3(p2)
        p3 = self.pool(c3)
        merge4 = self.pool(merge3)
        merge_c3 = p3 * merge4
        p3 = p3 + merge_c3

        c4 = self.conv4(p3)
        merge_c4 = c4 * merge4
        c4 = c4 + merge_c4
        
        up_5 = self.up(c4)
        merge5 = torch.cat([up_5, c3], dim=1)
        c5 = self.conv5(merge5)
        up_6 = self.up(c5)
        merge6 = torch.cat([up_6, c2], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up(c6)
        merge7 = torch.cat([up_7, c1], dim=1)
        c7 = self.conv7(merge7)
        c8 = self.conv8(c7)

        bottom = self.conv1_1(c4)

        return (c8, bottom), vessel_mask, c8, vessel_mask


if __name__ == '__main__':
    a = torch.ones((2,3,256,256))
    b = torch.ones((2,1,256,256))
    c = torch.ones((2,1,256,256))
    d = torch.ones((2,1,256,256))
    model = RNN_Model(3,1)
    predicts, targets, c10_seg, vessel_mask = model(a,b,c,d)
    # print(predicts.size())
    print(targets.size())
