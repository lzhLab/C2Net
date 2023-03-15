import torch
import torch.nn as nn
from torch.autograd import Variable


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        if torch.cuda.is_available():
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()) #[2,688,32]
        else:
            h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        output, _ = self.gru(x, h0) #[688,86,27],[2,688,32]->[688,86,32]
        return self.fc(output) 


if __name__ == '__main__':
    rnn = GRU(input_dim=27, hidden_dim=32, layer_dim=2, output_dim=1).cuda()
    x = torch.randn(8, 50, 27).cuda()
    y = rnn(x)
    print(y.shape)