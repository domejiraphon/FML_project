import os
import sys

def add_common_flags(parser):
  #training schedule
  parser.add_argument("--lr", type=float, default=1e-3,
              help="learning rate for model")
  parser.add_argument("--epochs", type=int, default=1000,
              help="Number of epoch to train")
  parser.add_argument("--print_every", type=int, default=1, 
              help="print loss every number of epoch")
  parser.add_argument("--save_every", type=int, default=2, 
              help="save model for every number of epoch")
  parser.add_argument("--batch_size", type=int, default=512, 
            help="batch size for model")

  #model
  parser.add_argument('--model_path', type=str, default="e1", 
            help = "the path to store model directory in runs")
  parser.add_argument('--restart', action='store_true',
            help = "restart experiment by deleting the old folder")
  parser.add_argument('--resnet', action='store_true',
            help = "Using resnet for a model")
  parser.add_argument("--dropout", type=float, default=2e-1,
              help="Dropout for model")
  #dataset
  parser.add_argument('--dataset_path', type=str, default="./data", 
            help = "the path that stores dataset")
  parser.add_argument("--num_classes", type=int, default=10, 
              help="Number of class")
  parser.add_argument("--num_workers", type=int, default=8, 
              help="Number of workers for dataloader")
  