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
from utils import model_utils
from utils import dataset_utils
from autoattack.autoattack import AutoAttack

import trainer 
import torchvision.models as models
import getpass

parser = argparse.ArgumentParser()
cmd_args_utils.add_common_flags(parser)
args = parser.parse_args()

def main():
  torch.manual_seed(0)
  np.random.seed(0) 
  args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  cfg = args 
  print(f"Train using: {cfg.device}")
  
  if not os.path.exists("./runs"):
    os.system("mkdir ./runs")
  if args.restart:
    os.system("rm -rf " + "runs/" + cfg.model_path)
  dataset, dataloader = dataset_utils.get_dataset(cfg)
  
  model, optimizer, scheduler = model_utils.get_network(cfg)
  adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard', verbose=False)
  runpath = "runs/"
  ckpt = runpath + cfg.model_path + "/ckpt.pt"
  start_epoch = 0
  if os.path.exists(ckpt):
    start_epoch = utils.loadFromCheckpoint(ckpt, model, optimizer, cfg)
  

  ts = utils.TrainingStatus(num_steps=args.epochs * dataloader['train'].__len__())
  writer = SummaryWriter(runpath + cfg.model_path)
  writer.add_text('command', ' '.join(sys.argv), 0)
  model.train()
  
  trainer.trainer(model=model,
          adversary=adversary,
          optimizer=optimizer,
          scheduler=scheduler,
          dataset = dataset, 
          dataloader = dataloader, 
          start_epoch = start_epoch,
          training_status = ts,
          step = start_epoch * dataloader['train'].__len__(),
          writer = writer,
          ckpt = ckpt,
          cfg = args)
  exit()   

if __name__ == "__main__":
    sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
    main()