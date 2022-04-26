import argparse
from autoaugment import *
from dataset_utils import *
from attack_utils import *
from loss_utils import _jensen_shannon_div, ALP
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim.lr_scheduler as lr_scheduler
#import torch.optim.lr_scheduler import OneCycleLR, MultiStepLR
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from autoattack import AutoAttack
from robustbench.data import load_cifar10
from utils import grad_norm
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import awp
import copy 

class CR_pl(LightningModule):
  def __init__(self, hparams, backbone, **kwargs):
    super().__init__()

    self.args = hparams
    self.hparams.update(vars(hparams))
    self.model = backbone
    self.criterion = nn.CrossEntropyLoss()
    self.kwargs = {'pin_memory': hparams.pin_memory, 'num_workers': hparams.num_workers}
    self.train_set, self.test_set, self.image_size, self.n_classes = get_dataset('autoaug', True)
    self.adversary = attack_module(self.model, self.criterion)
    if hparams.awp:
      proxy = copy.deepcopy(self.model).half().to("cuda")
      proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
      self.awp_adversary = awp.AdvWeightPerturb(proxy=proxy, 
                proxy_optim=proxy_opt, 
                gamma=1e-2,
                num_iter=hparams.num_iter)
    self.autoattack = AutoAttack(backbone, 
                  norm='Linf', eps=8/255, version='custom', 
                  attacks_to_run=['apgd-ce', 'apgd-dlr', 'apgd-t', 'fab-t'],
                  verbose=False,)
    self.autoattack.apgd.n_restarts = 1

  def forward(self, x):
    out = self.model(x)
    return out

  def _train(self, batch, stage, batch_idx=None, log=True):
    images, labels = batch
    
    images_aug1, images_aug2 = images[0], images[1]
    images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
    images_adv = self.adversary(images_pair, labels.repeat(2))
    if self.hparams.awp:
      self.awp = self.awp_adversary.calc_awp(model=self.model,
                                inputs_adv=images_adv,
                                targets=labels.repeat(2))
      self.awp_adversary.perturb(self.model, self.awp)
    loss = 0
    if self.hparams.alp:
      all_imgs = torch.cat([images_pair, images_adv], 0)
      outputs = self(all_imgs)
      outputs_imgs = outputs[:images_pair.shape[0]]
      outputs_adv = outputs[images_pair.shape[0]:]
      loss_alp = self.hparams.lam2 * ALP(images_pair, images_adv, outputs_imgs, outputs_adv)
      loss = loss + loss_alp 
    elif self.hparams.grad_norm:
      outputs_adv = grad_norm.normalize_gradient(self.model, images_adv)
    else:
      outputs_adv = self(images_adv)

    loss_ce = self.criterion(outputs_adv, labels.repeat(2))

      ### consistency regularization ###
    outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
    loss_con = self.hparams.lam * _jensen_shannon_div(outputs_adv1, outputs_adv2, self.hparams.T)

    ### total loss ###
    loss = loss + loss_ce + loss_con

    if log:
      if stage:
        self.log(f"{stage}/loss", loss, prog_bar=True)
        #robust_accuracy_dict = self.eval_autoattack()
        #print(robust_accuracy_dict)
        if batch_idx % 100 == 0:
          robust_accuracy_dict = self.eval_autoattack()
          for key, val in robust_accuracy_dict.items():
            self.log(f"Autoattack/{key}", val*100)
      
    return loss
  
  def _train2(self, batch, stage, batch_idx=None, log=True):
    images, labels = batch
    
    images_aug1, images_aug2 = images[0], images[1]
    images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
    images_adv = self.adversary(images_pair, labels.repeat(2))
    if self.hparams.awp:
      self.awp = self.awp_adversary.calc_awp(model=self.model,
                                inputs_adv=images_adv,
                                targets=labels.repeat(2))
    
      self.awp_adversary.perturb(self.model, self.awp)
    
      

    outputs_adv = self.model(images_adv)

    loss = self.criterion(outputs_adv, labels.repeat(2))


    if log:
      if stage:
        self.log(f"{stage}/loss", loss, prog_bar=True)
        #robust_accuracy_dict = self.eval_autoattack()
        #print(robust_accuracy_dict)
        if batch_idx % 100 == 0:
          robust_accuracy_dict = self.eval_autoattack()
          for key, val in robust_accuracy_dict.items():
            self.log(f"Autoattack/{key}", val*100)
      
    return loss
  
  def training_step(self, batch, batch_idx, log=True):
    loss = self._train(batch, "train", batch_idx, log)
    return loss 

  def evaluate(self, batch, stage=None, log=True):
    images, labels = batch
    images_aug1, images_aug2 = images[0], images[1]
    images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
    with torch.enable_grad():
      images_adv = self.adversary(images_pair, labels.repeat(2))

    outputs_adv = self(images_adv)
    loss_ce = self.criterion(outputs_adv, labels.repeat(2))

    ### consistency regularization ###
    outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
    loss_con = self.hparams.lam * _jensen_shannon_div(outputs_adv1, outputs_adv2, self.hparams.T)

    ### total loss ###
    loss = loss_ce + loss_con
    if log:
      if stage:
        self.log(f"{stage}_loss", loss, prog_bar=True)
    else:
      return loss 

  def validation_step(self, batch, batch_idx, log=True):
    self.evaluate(batch, "val", log=log)

  def test_step(self, batch, batch_idx, log=True):
    loss = self.evaluate(batch, "test", log=log)
    if not log:
      return loss

  def train_dataloader(self):
    trainloader = DataLoader(self.train_set, shuffle=True, batch_size=self.hparams.batch_size, **self.kwargs)
    # imgs, labels = next(iter(a))
    # print(f"aa: {imgs[0].shape}")
    return trainloader

  def val_dataloader(self):
    valloader = DataLoader(self.test_set, shuffle=False, batch_size=self.hparams.batch_size, **self.kwargs)
    return valloader

  def test_dataloader(self):
    testloader = DataLoader(self.test_set, shuffle=False, batch_size=self.hparams.batch_size, **self.kwargs)
    return testloader

  def configure_optimizers(self):
    if self.hparams.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=5e-4,
        )
    elif self.hparams.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        raise NotImplementedError()

    #steps_per_epoch = (int)(45000 // self.hparams.batch_size)
    lr_decay_gamma=0.1
    milestones = [int(0.5 * self.hparams.max_epochs), int(0.75 * self.hparams.max_epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    # scheduler_dict = {
    #     "scheduler": OneCycleLR(
    #         optimizer,
    #         0.1,
    #         epochs=self.trainer.max_epochs,
    #         steps_per_epoch=steps_per_epoch,
    #         ),
    #     "interval": "step",
    # }
    return {"optimizer": optimizer, "lr_scheduler": scheduler}
  
  def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
    """
    Parameters you define here will be available to your model through self.hparams
    :param parent_parser:
    :param root_dir:
    :return:
    """
    parser = argparse.ArgumentParser(parents=[parent_parser])

    # Model parameters
    parser.add_argument('--lam', default=1, type=float)
    parser.add_argument('--lam2', default=100, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_iter', default=1, type=int)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    return parser

  def eval_autoattack(self):
    self.model.eval()
    with torch.no_grad():
      x_test, y_test = load_cifar10(n_examples=1000)
      robust_accuracy_dict = self.autoattack.run_standard_evaluation(x_test.cuda(), y_test.cuda())
    self.model.train()
    return robust_accuracy_dict

  def training_step_end(self, batch_parts):
    if self.hparams.awp:
      self.awp_adversary.restore(self.model, self.awp)
    
      
  