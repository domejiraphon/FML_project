import torch 
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

class BasicBlock(nn.Module):
  def __init__(self, in_channels, out_channels, 
      kernel_size=3, activation_fn=None, dropout=None, **kwargs):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=in_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=1, padding=1,)
    self.bn1 = nn.BatchNorm2d(num_features=out_channels)
    self.dropout1 = nn.Dropout(dropout)
    self.conv2 = nn.Conv2d(in_channels=out_channels, 
                    out_channels=out_channels, 
                    kernel_size=kernel_size, 
                    stride=2, padding=1,)
    self.bn2 = nn.BatchNorm2d(num_features=out_channels)
    self.dropout2 = nn.Dropout(dropout)
    self.activation = activation_fn
    self.block = nn.Sequential(self.conv1, 
                    self.bn1,
                    self.activation,
                    self.dropout1,
                    self.conv2,
                    self.bn2, 
                    self.activation,
                    self.dropout2)
  def forward(self, x):
    return self.block(x)
    
class CNN(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    f_maps = [3, 8, 16, 32]
    input_size = 32
    activation_fn = nn.LeakyReLU()
    self.net = nn.ModuleList([
                  BasicBlock(f_maps[i], f_maps[i+1], activation_fn=activation_fn, dropout=cfg.dropout) 
                  for i in range(len(f_maps) - 1)])
    input_dims = int(input_size/(2**(len(f_maps) - 1)))**2 * f_maps[-1]
    
    head = [nn.Linear(input_dims, int(input_dims/2)),
            activation_fn,
            nn.Linear(int(input_dims/2), cfg.num_classes)]
    #head.append(self.activation)
    self.head = nn.Sequential(*head)
  def forward(self, x):
    out = x 
    for block in self.net:
      out = block(out)
    out = out.view(out.shape[0], -1)
    out = self.head(out)
    return out 
