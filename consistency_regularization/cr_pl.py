import argparse
from autoaugment import *
from dataset_utils import *
from attack_utils import *
from loss_utils import off_diagonal, _jensen_shannon_div, _H_min_div 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, MultiStepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from transformers import get_cosine_schedule_with_warmup
import copy 
import awp 
import pandas as pd 

class CR_pl(LightningModule):
  def __init__(self, hparams, backbone, proxy=None):
    super().__init__()

    self.args = hparams
    self.hparams.update(vars(hparams))
    
    # setup model
    self.model = backbone
    if hparams.extra_reg != None:
        self.projector = self.Projector(hparams.embed_dim)
    
    # setup criterion
    self.criterion = nn.CrossEntropyLoss()
    if proxy:
      proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
      self.awp_adversary = awp.AdvWeightPerturb(proxy=proxy, 
                    proxy_optim=proxy_opt, 
                    gamma=1e-2)
    # setup dataset and adversary attack
    self.kwargs = {'pin_memory': hparams.pin_memory, 'num_workers': hparams.num_workers}
    self.train_set, self.test_set, self.image_size, self.n_classes = get_dataset('autoaug', True)
    self.adversary = attack_module(self.model, self.criterion)
    self.log_dict = {"step": [], "loss ce": [], "loss con": []}

  def Projector(self, embedding):
    mlp_spec = f"{embedding}-{self.hparams.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.model(x)
    return out

  def _forward(self, batch, stage):
    images, labels = batch
    images_aug1, images_aug2 = images[0], images[1]
    images_pair = torch.cat([images_aug1, images_aug2], dim=0)  # 2B
   
    #print(images_pair.shape) 
    
    if stage == "train":
        images_adv = self.adversary(images_pair, labels.repeat(2))
    else:
        with torch.enable_grad():
            images_adv = self.adversary(images_pair, labels.repeat(2))
    if stage == "train" and self.hparams.awp:
      self.awp = self.awp_adversary.calc_awp(model=self.model,
                                inputs_adv=images_adv,
                                targets=labels.repeat(2))
      self.awp_adversary.perturb(self.model, self.awp)
    # register hook to get intermediate output
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook
    self.model.relu.register_forward_hook(get_activation('penultimate'))
    # get original output
    outputs_adv = self.model(images_adv)
    # get latent
    outputs_latent = activation['penultimate']
    outputs_latent = F.avg_pool2d(outputs_latent, 8)
    outputs_latent = outputs_latent.view(outputs_latent.size(0), -1)
    #print(f"outputs_adv: {outputs_adv.shape}")
    #print(f"outputs_latent: {outputs_latent.shape}")
    loss_ce = self.criterion(outputs_adv, labels.repeat(2))

    ### consistency regularization ###
    outputs_adv1, outputs_adv2 = outputs_adv.chunk(2)
    #print(f"outputs_adv1.shape: {outputs_adv1.shape}, outputs_adv2.shape: {outputs_adv2.shape}")
    if self.hparams.loss_func == 'JS':
        loss_con = self.hparams.con_coeff * _jensen_shannon_div(outputs_adv1, outputs_adv2, self.hparams.T)
    elif self.hparams.loss_func == 'MIN':
        loss_con = self.hparams.con_coeff * _H_min_div(outputs_adv1, outputs_adv2, self.hparams.T) 
    else:
        raise NotImplementedError() 

    # calculate covariance regularization by projecting laten
    # representation with a given projector
    if self.hparams.extra_reg == 'cov':
        num_features = int(self.hparams.mlp.split("-")[-1])
        #self.projector = self.Projector(outputs_latent.shape[1])
        outputs_latent1, outputs_latent2 = outputs_latent.chunk(2)
        outputs_latent1, outputs_latent2 = self.projector(outputs_latent1), self.projector(outputs_latent2)
        cov_1 = (outputs_latent1.T @ outputs_latent1) / (self.hparams.batch_size - 1)
        cov_2 = (outputs_latent2.T @ outputs_latent2) / (self.hparams.batch_size - 1)
        loss_cov = off_diagonal(cov_1).pow_(2).sum().div(num_features) + \
                off_diagonal(cov_2).pow_(2).sum().div(num_features)
        #print(f"loss_cov before: {loss_cov}") 
        loss_cov *= self.hparams.cov_coeff
        #print(f"loss_cov after: {loss_cov}")
        #print(f"loss_con: {loss_con}")
        if stage:
            self.log(f"{stage}_loss_cov", loss_cov, prog_bar=True)
    else:
        loss_cov = 0
    ### total loss ###
    loss_ce *= self.hparams.sim_coeff
    loss = loss_ce + loss_con + loss_cov

    if stage:
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_loss_con", loss_con, prog_bar=True)
        self.log(f"{stage}_loss_sim", loss_ce, prog_bar=True)
    if stage == "val":
      self.log_info(loss_ce, loss_con)
      
    return loss
   

  def training_step(self, batch, batch_idx):
    loss = self._forward(batch, "train")
    return loss

  def validation_step(self, batch, batch_idx):
    self.log_each_val = {"loss ce": [], "loss con": []}
    self._forward(batch, "val")
    
  def test_step(self, batch, batch_idx):
    self._forward(batch, "test")

  def train_dataloader(self):
    trainloader = DataLoader(self.train_set, shuffle=True, batch_size=self.hparams.batch_size, **self.kwargs)
    self.train_len = len(trainloader)
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
    
    tb_size = self.hparams.batch_size * max(1, self.trainer.num_devices)
    ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
    self.total_steps = (len(self.train_set) // tb_size) // ab_size
    self.warmup_steps = 0.06 * self.total_steps
    
    #steps_per_epoch = (int)(45000 // self.hparams.batch_size)
    if self.hparams.scheduler=="multistep":
        lr_decay_gamma = 0.1
        milestones = [int(0.5 * self.hparams.max_epochs), int(0.75 * self.hparams.max_epochs)]
        scheduler = MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
    elif self.hparams.scheduler=="cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.warmup_steps,
                                                    num_training_steps=self.total_steps,
                                                )
    else:
        raise NotImplementedError()
    
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
    parser.add_argument('--loss_func', choices=["JS", "MIN"], default="JS", type=str)
    parser.add_argument('--sim_coeff', default=1.0, type=float) 
    parser.add_argument('--con_coeff', default=1.0, type=float)
    parser.add_argument('--cov_coeff', default=1.0, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--embed_dim', default=640, type=int)
    parser.add_argument("--mlp", default="2048-2048-2048", type=str, help='Size and number of layers of the MLP expander head')
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--extra_reg", choices=["cov"], type=str, default=None)
    parser.add_argument('--optimizer', choices=["adamw", "sgd"], default="sgd", type=str)
    parser.add_argument('--scheduler', choices=["cosine", "multistep"], default="multistep", type=str)
    parser.add_argument('--warmup', default=False, type=bool)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    return parser

  def training_step_end(self, batch_parts):
    if self.hparams.awp:
      self.awp_adversary.restore(self.model, self.awp)

  
  def validation_epoch_end(self, batch_parts):

    mean_ce = torch.tensor(self.log_each_val["loss ce"]).mean()
    mean_con = torch.tensor(self.log_each_val["loss con"]).mean()

    self.log_dict["step"].append(self.global_step)
    self.log_dict["loss ce"].append(mean_ce.item())
    self.log_dict["loss con"].append(mean_con.item())

  def log_info(self, loss_ce=None, loss_con=None, write_pd=False, dir=None):
    if write_pd:
      df = pd.DataFrame.from_dict(self.log_dict)
      df.to_csv(os.path.join(dir, 'loss.csv'), index=False)
      return 
  
    self.log_each_val["loss ce"].append(loss_ce)
    self.log_each_val["loss con"].append(loss_con)
    
    
