import cr_pl
from cr_pl import *
from vit_pytorch import ViT
import WideResNet
from WideResNet import *
from robustbench.data import load_cifar10
from autoattack import AutoAttack
import pandas as pd 
import os, sys, glob 
import utils
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
    if hparams.backbone_model == 'WResNet':
        backbone = WideResNet(depth=hparams.net_depth, 
                         n_classes=hparams.num_classes, 
                         widen_factor=hparams.wide_factor,
                         use_sn=hparams.use_sn)
    elif hparams.backbone_model == 'ViT':
        backbone = ViT(image_size=32, patch_size=4, num_classes=hparams.num_classes, dim=768, depth=12,
                    heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1, pool='mean')
    else:
        raise NotImplementedError()
    #backbone = create_model()
    model = CR_pl(hparams, backbone)

    #x_test, y_test = load_cifar10(n_examples=1000)
    x_test, y_test = load_cifar10()
    print("Test mode")
    # Test the model from loaded checkpoint
    ckpt = sorted(glob.glob(os.path.join(hparams.runpath, hparams.model_dir, "*.ckpt")))
    if hparams.use_swa:
      ckpt = ckpt[-1]
    else:
      ckpt = ckpt[0]
    print(f"Load from: {ckpt}")
    checkpoint = torch.load(ckpt)
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.model.load_state_dict(state_dict, strict=True)

    backbone = model.model.cuda()
    backbone.eval()
    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    #adversary = AutoAttack(backbone, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    robust_accuracy_dict = adversary.run_standard_evaluation(x_test.cuda(), y_test.cuda())
    """
    for k, v in robust_accuracy_dict.items():
      robust_accuracy_dict[k] = v.detach().cpu().numpy() * 100
    df = pd.DataFrame.from_dict(robust_accuracy_dict, orient="index")
    df.to_csv(os.path.join(hparams.runpath, hparams.model_dir, "metrics.csv"))
    """
if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    sys.excepthook = utils.colored_hook(os.path.dirname(os.path.realpath(__file__)))
    root_dir = os.path.dirname('./cr_pl')
    parent_parser = argparse.ArgumentParser(add_help=False)

    # gpu args
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
        '--backbone_model',
        choices=["WResNet", "ViT"],
        type=str,
        default="WResNet",
        help='backbone model'
    )

    parent_parser.add_argument(
        '--net_depth',
        type=int,
        default=28,
        help='wide resnet depth'
    )

    parent_parser.add_argument(
        '--wide_factor',
        type=int,
        default=4,
        help='wide resenet wide factor'
    )

    parent_parser.add_argument(
        '--load_path',
        type=str,
        default=None,
        help='load path for saved model'
    )

    parent_parser.add_argument(
        '--use_sn', 
        action="store_true", 
        help="Use Spectral Normalization",
    )

    parent_parser.add_argument(
        '--use_awp', 
        action="store_true", 
        help="Use weight perturbation",
    )

    parent_parser.add_argument(
        '--use_swa', 
        action="store_true", 
        help="Use Stochastic weighted average",
    )

    parent_parser.add_argument(
        '--runpath', 
        type=str, 
        default="./runs", 
        help = "the path to store all models"
    )

    parent_parser.add_argument(
        '--restart', 
        action="store_true", 
        help = "restart the runs"
    )

    parent_parser.add_argument(
        '--model_dir', 
        type=str,
        default="e1", 
        help = "the path to store model directory"
    )
    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)