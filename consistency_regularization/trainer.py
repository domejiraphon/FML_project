import numpy as np
import os
import sys
import argparse
import torch
import torch.nn as nn
from utils import utils 
import copy 
import torch.nn.functional as F 

#from autoattack import AutoAttack
def eval_test(model, writer, testdataloader, hparams):
  loss = 0
  for batch_idx, batch in enumerate(testdataloader):
    if hparams.device == torch.device("cuda"):
      batch = ((batch[0][0].half().to(hparams.device), 
                  batch[0][1].half().to(hparams.device)),
                  batch[1].to(hparams.device))
    loss += model.test_step(batch, batch_idx, log=False)
  writer.add_scalar('test/loss', loss/len(model.test_dataloader()), step)

def trainer(model, 
            optimizer, 
            writer,
            hparams, 
            ckpt,
            start_epoch,
            ts,
            step,
            **kwargs):
  trainloader = model.train_dataloader()
  testdataloader = model.test_dataloader()
  optim = optimizer["optimizer"]
  sched = optimizer["lr_scheduler"]
  for epoch in range(start_epoch, hparams.max_epochs+1):
    for batch_idx, batch in enumerate(trainloader):
      ts.tic()
    
      if hparams.device == torch.device("cuda"):
        batch = ((batch[0][0].to(hparams.device), 
                  batch[0][1].to(hparams.device)),
                  batch[1].to(hparams.device))
     
      loss = model.training_step(batch, batch_idx, log=False)
      log = ts.toc(step, loss.item())
      print(log)
      optim.zero_grad()
      loss.backward()
      optim.step()
      sched.step()
      step += 1
      if step % 50 == 0:
        writer.add_scalar('train/loss', loss, step)
      
        with torch.no_grad():
          robust_accuracy_dict = model.eval_autoattack()
          for key, val in robust_accuracy_dict.items():
            writer.add_scalar(f"Autoattack/{key}", val*100)
    with torch.no_grad():
      eval_test(model, writer, testdataloader, hparams)
      