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
  parser.add_argument('--vgg', action='store_true',
            help = "Using vgg for a model")
  parser.add_argument('--densenet', action='store_true',
            help = "Using densenet for a model")
  parser.add_argument('--dla', action='store_true',
            help = "Using DLA for a model")
  parser.add_argument("--dropout", type=float, default=0.0,
            help="Dropout for model")

  #dataset
  parser.add_argument('--attack', action='store_true',
            help = "To train with adversarial attack")
  parser.add_argument('--dataset_path', type=str, default="./data", 
            help = "the path that stores dataset")
  parser.add_argument("--num_classes", type=int, default=10, 
            help="Number of class")
  parser.add_argument("--num_workers", type=int, default=8, 
            help="Number of workers for dataloader")

  #adversarial
  parser.add_argument('--attack_method', type=str, default='pgd', 
              help='attacking method: pgd | mifgsm')
  parser.add_argument('--epsilon', type=float, default=0.3, 
              help='the maximum allowed perturbation per pixel')
  parser.add_argument('--inner_loop', type=int, default=5, 
              help='the number of PGD iterations used by the adversary')
  parser.add_argument('--alpha', type=float, default=0.01, 
              help='the size of the PGD adversary steps')
  parser.add_argument('--mu', type=float,  default=1.0, 
              help='Moment for MIGFSM method')
  parser.add_argument('--random_start', action='store_false', 
              help='if random start')
  parser.set_defaults(resnet=True)
  