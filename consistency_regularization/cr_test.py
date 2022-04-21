import cr_pl
from cr_pl import *
import WideResNet
from WideResNet import *
from robustbench.data import load_cifar10
from autoattack import AutoAttack

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
    backbone = WideResNet(depth=hparams.net_depth, n_classes=hparams.num_classes, widen_factor=hparams.wide_factor)
    #backbone = create_model()
    model = CR_pl(hparams, backbone)

    x_test, y_test = load_cifar10(n_examples=100)
    print("Test mode")
    # Test the model from loaded checkpoint
    checkpoint = torch.load(hparams.load_path)
    model.load_state_dict(checkpoint['state_dict'])
    backbone = model.model.cuda()
    backbone.eval()
    #adversary = AutoAttack(model, norm='Linf', eps=8/255)
    adversary = AutoAttack(backbone, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    x_adv = adversary.run_standard_evaluation(x_test.cuda(), y_test.cuda())

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

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
        '--load_path',
        type=str,
        default=None,
        help='load path for saved model'
    )

    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
