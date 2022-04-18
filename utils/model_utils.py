import torch 
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
ce_loss = lambda one_label, out, softmax : torch.mean(- torch.sum(one_label * torch.log(softmax(out) + 1e-12), dim = 1))

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

def evaluate(model, 
            dataloader, 
            writer,
            cfg,
            epoch):
 
  softmax = nn.Softmax(dim=-1)
  correct, num, all_loss = 0.0, 0.0, 0.0
  for _, features in enumerate(dataloader):
    imgs = features["img"].to(cfg.device)
    labels = features["labels"].to(cfg.device)
    
    out = model(imgs)
    one_hot_label = F.one_hot(labels,  num_classes = cfg.num_classes)
    loss = ce_loss(one_hot_label, out, softmax)
  
    pred = out.data.max(1, keepdim=True)[1]
    correct += pred.eq(labels.data.view_as(pred)).sum().item()
    num += labels.shape[0]
    all_loss += loss 

  acc = correct / num * 100.0
  all_loss = all_loss / num 

  writer.add_scalar('test/loss', all_loss, epoch)
  writer.add_scalar('test/Acc', acc, epoch)
  return {"Accuracy": acc,
          "Loss": all_loss}
  
def get_network(cfg):
  if cfg.resnet:
    model = models.resnet18(pretrained=False) 
  else:
    model = CNN(cfg)
 
  model = model.to(cfg.device)
  optimizer = torch.optim.Adam(model.parameters(), 
                        lr=cfg.lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                T_max=cfg.epochs)
  return model, optimizer, scheduler