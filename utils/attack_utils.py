import torch 
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from utils import model_utils

class LinfPGDAttack(object):
  """ 
      Attack parameter initialization. The attack performs k steps of size 
      alpha, while always staying within epsilon from the initial point.
          IFGSM(Iterative Fast Gradient Sign Method) is essentially 
          PGD(Projected Gradient Descent) 
  """
  def __init__(self, 
              model, 
              epsilon=0.3, 
              inner_loop=40, 
              alpha=0.01, 
              random_start=True,
              num_classes=-1):
    self.model = model
    self.epsilon = epsilon
    self.inner_loop = inner_loop
    self.alpha = alpha
    self.random_start = random_start
    self.softmax = nn.Softmax(dim=-1)
    self.num_classes = num_classes

  def forward(self, imgs, labels):
    if self.random_start:
      imgs_adv = imgs + imgs.new(imgs.size()).uniform_(-self.epsilon, self.epsilon)
    else:
      imgs_adv = imgs.clone()
    training = self.model.training
    if training:
      self.model.eval()
    for _ in range(self.inner_loop):
      imgs_adv.requires_grad = True 

      out = self.model(imgs_adv)
      one_hot_label = F.one_hot(labels, num_classes=self.num_classes)
      loss = model_utils.ce_loss(one_hot_label, out, self.softmax)
      loss.backward()

      grad = imgs_adv.grad.clone()
      # update x_adv
      imgs_adv = imgs_adv.detach() + self.alpha * grad.sign()
      
      # x_adv = np.clip(x_adv, x_adv-self.epsilon, x_adv+self.epsilon)
      imgs_adv = torch.min(torch.max(imgs_adv, imgs - self.epsilon), imgs + self.epsilon)
      #imgs_adv = torch.clamp(imgs_adv, imgs - self.epsilon, imgs + self.epsilon)
      
      imgs_adv.clamp_(0, 1)

    if training:
      self.model.train()
    return imgs_adv
    