import numpy as np
import os
import sys
import argparse
import torch
import torch.nn as nn
from utils import utils 
import copy 
import torch.nn.functional as F 

from utils import utils 
from utils import model_utils 
#from autoattack import AutoAttack

def trainer(model, 
            attack,
            adversary, 
            optimizer, 
            dataset, 
            dataloader, 
            writer,
            cfg, 
            training_status, 
            ckpt,
            step,
            start_epoch,
            **kwargs):
 
  softmax = nn.Softmax(dim=-1)
 
  for epoch in range(start_epoch, cfg.epochs+1):
    correct, num = 0.0, 0.0
    for _, features in enumerate(dataloader['train']):
      training_status.tic()
      
      imgs = features["img"].to(cfg.device)
      labels = features["labels"].to(cfg.device)
      one_hot_label = F.one_hot(labels,  num_classes = cfg.num_classes)
      if attack:
        imgs_adv = attack.forward(imgs, labels)
        imgs = torch.cat([imgs, imgs_adv], 0)
        #one_hot_label = one_hot_label.repeat([2, 1])
        #labels = labels.repeat([2])
       
      out = model(imgs)
      clean_out = out[:imgs_adv.shape[0]]
      clean_loss = model_utils.ce_loss(one_hot_label, clean_out, softmax)

      adv_out = out[imgs_adv.shape[0]:]
      adv_loss = model_utils.ce_loss(one_hot_label, adv_out, softmax)

      loss = clean_loss + adv_loss
      log = training_status.toc(step, loss.item())
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      pred = clean_out.data.max(1, keepdim=True)[1]
      correct += pred.eq(labels.data.view_as(pred)).sum().item()

      num += labels.shape[0]
      step += 1
    acc = correct / num * 100.0
    if epoch % cfg.print_every == 0:
      print(log)
      writer.add_scalar('train/loss', loss, epoch)
      writer.add_scalar('train/Acc', acc, epoch)
      #writer.add_scalar('Train/loss', loss, epoch)
      with torch.no_grad():
        metrics = model_utils.evaluate(model, 
                      dataloader=dataloader['test'], 
                      cfg=cfg,
                      writer=writer,
                      epoch=epoch)
    """
    if epoch % 10 == 0 and epoch != 0:  
      for typ in ["train", "test"]:
        robust_accuracy_dict = adversary.run_standard_evaluation(dataset[typ].data, 
                dataset[typ].targets, bs=cfg.batch_size)
        
        for key, val in robust_accuracy_dict.items():
          writer.add_scalar(f'{typ}/{key}', val * 100, epoch)
    """
    #if (epoch % cfg.save_every == 0 and epoch != 0) or epoch == cfg.epochs:
    #  utils.checkpoint(ckpt, model, optimizer, epoch+1)
  for typ in ["train", "test"]:
    adversary.run_standard_evaluation(dataset[typ].data, 
                dataset[typ].targets, bs=cfg.batch_size)