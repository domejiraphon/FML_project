import os
import sys

def add_common_flags(parent_parser):
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--num_nodes',
        type=int,
        default=1,
        help='how many nodes'
    )
    parent_parser.add_argument(
        '--precision',
        type=int,
        default=16,
        help='default to use mixed precision 16'
    )

    parent_parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='number of classes for the dataset'
    )

    parent_parser.add_argument(
        '--net_depth',
        type=int,
        default=34,
        help='wide resnet depth'
    )

    parent_parser.add_argument(
        '--wide_factor',
        type=int,
        default=10,
        help='wide resenet wide factor'
    )
    parent_parser.add_argument(
        '--use_sn',
        action="store_true",
        help="Use spectral Normalization",
    )
    parent_parser.add_argument(
        '--restart',
        action="store_true",
        help="Remove tensorboard and ckpt",
    )
    
    parent_parser.add_argument(
        '--alp',
        action="store_true",
        help="Use ALP",
    )
    #model
    parent_parser.add_argument('--grad_norm', action="store_true", 
            help="Use grad_norm",)
    parent_parser.add_argument('--runpath', type=str, default="./runs", 
            help = "the path to store all models")
    parent_parser.add_argument('--model_dir', type=str, default="e1", 
            help = "the path to store model directory")
  
  