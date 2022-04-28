import cr_pl
from cr_pl import *
from vit_pytorch import ViT
import WideResNet
from WideResNet import *
from robustbench.data import load_cifar10
from autoattack import AutoAttack

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class EnsembledModels(nn.Module):
    def __init__(self, models):
      super(EnsembledModels, self).__init__()
      self.model_lst = models

    def forward(self, x):
      logits = None
      for idx, model in enumerate(self.model_lst):
        if idx == 0:
          logits = model(x).cuda()
        else:
          logits += model(x).cuda()
      logits /= len(self.model_lst)
      return logits

def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    if hparams.ensemble == False:
        if hparams.backbone_model == 'WResNet':
            backbone = WideResNet(depth=hparams.net_depth, n_classes=hparams.num_classes, widen_factor=hparams.wide_factor)
        elif hparams.backbone_model == 'ViT':
            backbone = ViT(image_size=32, patch_size=4, num_classes=hparams.num_classes, dim=768, depth=12,
                        heads=16, mlp_dim=1024, dropout=0.1, emb_dropout=0.1, pool='mean')
        else:
            raise NotImplementedError()
    else:
        if hparams.backbone_model == 'WResNet':
            backbone1 = WideResNet(depth=hparams.net_depth, n_classes=hparams.num_classes, widen_factor=hparams.wide_factor)
            backbone2 = WideResNet(depth=hparams.net_depth, n_classes=hparams.num_classes, widen_factor=hparams.wide_factor) 
        else:
            raise NotImplementedError()
    #backbone = create_model()
    x_test, y_test = load_cifar10(n_examples=1000)
    print("Test Mode")
    if hparams.ensemble == False:
        model = CR_pl(hparams, backbone)
        # Test the model from loaded checkpoint
        checkpoint = torch.load(hparams.load_path1)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        backbone = model.model.cuda()
        backbone.eval()
    else:
        model1 = CR_pl(hparams, backbone1)
        model2 = CR_pl(hparams, backbone2)
        checkpoint1 = torch.load(hparams.load_path1)
        checkpoint2 = torch.load(hparams.load_path2)
        model1.load_state_dict(checkpoint1['state_dict'], strict=False)
        model2.load_state_dict(checkpoint2['state_dict'], strict=False)
        backbone1 = model1.model.cuda().eval()
        backbone2 = model2.model.cuda().eval()
        model_lst = [backbone1, backbone2]
        model = EnsembledModels(model_lst)

    adversary = AutoAttack(model, norm='Linf', eps=8/255)
    #adversary = AutoAttack(backbone, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
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
        '--backbone_model',
        choices=["WResNet", "ViT"],
        type=str,
        default="WResNet",
        help='backbone model'
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
        '--load_path1',
        type=str,
        default=None,
        help='load path 1 for saved model'
    )

    parent_parser.add_argument(
        '--load_path2',
        type=str,
        default=None,
        help='load path 2 for saved model'
    )

    parent_parser.add_argument(
        '--ensemble',
        action='store_true',
        help='decide if ensembling the models, if not, load_path1 is used for model loading'
    )

    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)

