import numpy as np
import os
import sys
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import getpass
from torch import nn

from utils import utils
from utils import cmd_args_utils

from autoattack.autoattack import AutoAttack

import trainer 
import torchvision.models as models
import getpass
import os, sys 
from WideResNet import *
import pytorch_lightning as pl
import torchvision
from WideResNet import *
from torch.utils.tensorboard import SummaryWriter
from cr_pl import *
from dataset_utils import * 
import numpy as np 
from pytorch_lightning.utilities.seed import seed_everything
SEED = 2022
seed_everything(SEED)
parser = argparse.ArgumentParser()
cmd_args_utils.add_common_flags(parser)
args = parser.parse_args()

def main(hparams):
  torch.manual_seed(0)
  np.random.seed(0) 
  
  if not os.path.exists("./runs"):
    os.system("mkdir ./runs")
  if args.restart:
    os.system("rm -rf " + "runs/" + hparams.model_dir)
 
  backbone = WideResNet(depth=hparams.net_depth, 
                         n_classes=hparams.num_classes, 
                         widen_factor=hparams.wide_factor,
                         use_sn=hparams.use_sn)
  #if hparams.precision == 16:
  #  backbone = backbone.half()
  backbone = backbone.to(hparams.device)

  cr_pl = CR_pl(hparams, backbone)

  runpath = "runs/"
  ckpt = runpath + hparams.model_dir + "/ckpt.pt"
  start_epoch = 0
  if os.path.exists(ckpt):
    start_epoch = utils.loadFromCheckpoint(ckpt, model, optimizer, hparams)
  
  ts = utils.TrainingStatus(num_steps=hparams.max_epochs* len(cr_pl.train_set))
  writer = SummaryWriter(runpath + hparams.model_dir)
  writer.add_text('command', ' '.join(sys.argv), 0)
  cr_pl.model.train()
  print(cr_pl.model)
  num_param = torch.tensor([param.numel() for param in cr_pl.model.parameters()]).sum()
  print(f"Number of model parameters: {num_param}")
 
  trainer.trainer(model=cr_pl,
          optimizer=cr_pl.configure_optimizers(),
          start_epoch=start_epoch,
          writer=writer,
          ckpt=ckpt,
          hparams=hparams,
          step = start_epoch * len(cr_pl.train_set),
          ts=ts)
  exit()   

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
    root_dir = os.path.dirname('./cr_pl')
    parent_parser = argparse.ArgumentParser(add_help=False)
    cmd_args_utils.add_common_flags(parent_parser)
    # gpu args
    
   
    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    hyperparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
