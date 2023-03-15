import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DB_Criterion(nn.Module):

    def __init__(self):
        super(DB_Criterion, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, predicts, targets):
        pred_1, pred_2 = predicts

        targets_2 = F.adaptive_max_pool2d(targets, pred_2.size()[2])
        loss1 = self.criterion(pred_1, targets)
        loss2 = self.criterion(pred_2, targets_2)

        loss = loss1 + loss2
        return loss / 2