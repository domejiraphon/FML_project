import torch 
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from utils.models.resnet import ResNet18
from utils.models.dla import DLA
from utils.models.vgg import VGG 
from utils.models.dense_net import densenet_cifar
from utils.models.custom_net import CNN

ce_loss = lambda one_label, out, softmax : torch.mean(- torch.sum(one_label * torch.log(softmax(out) + 1e-12), dim = 1))

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
    model = ResNet18()
  elif cfg.dla:
    model = Models.dla.DLA()
  elif cfg.vgg:
    model = Models.vgg.VGG('VGG16')
  elif cfg.vgg:
    model = Models.dense_net.densenet_cifar()
  else:
    model =Models.custom_net.CNN(cfg)
  
  model = model.to(cfg.device)
  optimizer = torch.optim.Adam(model.parameters(), 
                        lr=cfg.lr)
  #optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, 
  #                      momentum=0.9, weight_decay=0.0005)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                T_max=cfg.epochs)
  return model, optimizer, scheduler