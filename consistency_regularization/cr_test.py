import sys 
import os 
import glob 
import pandas as pd

import cr_pl
from cr_pl import *
import WideResNet
from WideResNet import *
from robustbench.data import load_cifar10
from autoattack import AutoAttack
from utils import utils 
from utils import cmd_args_utils
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    backbone = WideResNet(depth=hparams.net_depth, 
                         n_classes=hparams.num_classes, 
                         widen_factor=hparams.wide_factor,
                         use_sn=hparams.use_sn)
    #backbone = create_model()
    model = CR_pl(hparams, backbone)

    x_test, y_test = load_cifar10()
    print("Test mode")
    # Test the model from loaded checkpoint
    runpath = "./runs"
    ckpt = glob.glob(os.path.join(runpath, hparams.model_dir, "*.ckpt"))[0]
  
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['state_dict'])
    backbone = model.model.cuda()
    backbone.eval()
    #adversary = AutoAttack(model, norm='Linf', eps=8/255)
    adversary = AutoAttack(backbone, norm='Linf', eps=8/255, version='standard')
    adversary.apgd.n_restarts = 1
    robust_accuracy_dict = adversary.run_standard_evaluation(x_test.cuda(), y_test.cuda())
    df = pd.DataFrame.from_dict(robust_accuracy_dict, orient="index")
    df.to_csv(os.path.join(runpath, hparams.model_dir, "metrics.csv"))

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
    root_dir = os.path.dirname('./cr_pl')
    parent_parser = argparse.ArgumentParser(add_help=False)
    cmd_args_utils.add_common_flags(parent_parser)

    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
