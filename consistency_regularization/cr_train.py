import os, sys 
import WideResNet
from WideResNet import *
from cr_pl import *
from vit_pytorch import ViT
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import StochasticWeightAveraging, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger 
import utils
import glob 

import matplotlib.pyplot as plt 
SEED = 2022
#seed_everything(SEED)
import warnings
warnings.filterwarnings("ignore")
def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

def check_minima(cr_pl, hparams, perturb, range_x=0.2, n_pts=20, num_rand=None):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  ckpt = sorted(glob.glob(os.path.join(hparams.runpath, hparams.model_dir, "*.ckpt")))

  ckpt = ckpt[0]
  if hparams.use_swa:
    ckpt = "/".join(ckpt.split("/")[:-1]) +"/swa.ckpt"


  print(f"Load from: {ckpt}")
  checkpoint = torch.load(ckpt, map_location=device)
 
  state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
  cr_pl.model.load_state_dict(state_dict, strict=True)

  cr_pl.model.eval()
  vec_lenx = torch.linspace(-range_x, 0, int(n_pts/2))
  vec_lenx = torch.cat([vec_lenx, torch.linspace(0, range_x, int(n_pts/2)+1)[1:]], 0)
  v1 = utils.flatten(cr_pl.model.parameters())
  loss_surf = torch.zeros(num_rand, n_pts).to(device)
  perturb = perturb.to(device)

  train_loader = cr_pl.train_dataloader()
  start_pars = cr_pl.model.state_dict()

  with torch.no_grad():
    for ii in range(num_rand):
    
      perturb_dir = utils.unflatten_like(perturb[ii: ii+1], cr_pl.model.parameters())
      for jj in range(n_pts):
        for i, par in enumerate(cr_pl.model.parameters()):
          par.data = par.data + vec_lenx[jj] * perturb_dir[i]
        loss = 0
        for i, batch in enumerate(train_loader):
          loss += cr_pl._forward(batch, stage="eval")
          if i == 0: break

        loss_surf[ii, jj] = (loss / (i+1))
        cr_pl.model.load_state_dict(start_pars)
        print(f"Loss at {ii, jj}: {loss_surf[ii, jj]}")
  loss_surf = torch.mean(loss_surf, dim=0)
  return vec_lenx.cpu().numpy(), loss_surf.cpu().numpy()

def plot(vec_lenx, loss_surf, hparams):
  
  plt.plot(vec_lenx, loss_surf[0], 'r')
  plt.plot(vec_lenx, loss_surf[1], 'b')
  plt.grid()
  plt.xlabel("Model perturbation")
  plt.ylabel("Loss (log scale)")
  plt.legend(["Baseline", hparams.model_dir])
  plt.yscale("log")
  name = os.path.join(hparams.runpath, hparams.model_dir, "loss.jpg")
  plt.savefig(name)
  plt.clf()


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
    cr_pl = CR_pl(hparams, backbone)
    if hparams.infer:
      model_dir = ["baseline", hparams.model_dir]
      num_rand = 1
      v1 = utils.flatten(cr_pl.model.parameters())
      perturb = torch.randn((num_rand, v1.shape[0]))
      loss_surf = []
      for dir in model_dir:
        hparams.model_dir = dir
        if dir == "baseline":
       
          hparams.use_swa = False
          hparams.use_sn = False
          
        backbone = WideResNet(depth=hparams.net_depth, 
                          n_classes=hparams.num_classes, 
                          widen_factor=hparams.wide_factor,
                          use_sn=hparams.use_sn)
        cr_pl = CR_pl(hparams, backbone)
        vec_lenx, surf = check_minima(cr_pl, hparams, perturb, range_x=0.2, n_pts=20, num_rand=num_rand)
        loss_surf.append(surf)

      plot(vec_lenx, loss_surf, hparams)
      exit()
    #torch.autograd.set_detect_anomaly(True)
    # ------------------------
    # 2 DEFINE CALLBACKS
    # ------------------------
    bar = TQDMProgressBar(refresh_rate=1, process_position=0)
    if hparams.use_swa:
        swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, 
                                                annealing_strategy='cos', avg_fn=None)
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(hparams.runpath, hparams.model_dir), 
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_weights_only=False
        )
    #early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    
    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    callbacks = [checkpoint_callback, bar]
    if hparams.use_swa:
        callbacks.append(swa_callback)
    logger = TensorBoardLogger(save_dir=hparams.runpath,
                    name=hparams.model_dir,
                  )
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        gpus=hparams.gpus,
        precision=hparams.precision,
        num_nodes=hparams.num_nodes,
        max_epochs=hparams.max_epochs,
        #strategy=None,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=callbacks,
        logger=logger,
        )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(cr_pl)
    if hparams.use_swa:
      path = os.path.join(hparams.runpath, hparams.model_dir, "swa.ckpt")
      trainer.save_checkpoint(path)

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
        choices=["ViT", "WResNet"],
        default="WResNet", 
        type=str,
        help='name of backbone model'
    )

    parent_parser.add_argument(
        '--use_sn', 
        action='store_true',
        help="Use Spectral Normalization",
    )

    parent_parser.add_argument(
        '--use_awp', 
        action='store_true', 
        help="Use Stochastic Weight Perturbation",
    )

    parent_parser.add_argument(
        '--use_swa', 
        action='store_true',
        help="Use Stochastic Weighted Average",
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
        '--runpath', 
        type=str, 
        default="./runs", 
        help = "the path to store all models"
    )

    parent_parser.add_argument(
        '--model_dir', 
        type=str,
        default="e1",
        help = "the path to store model directory"
    )

    parent_parser.add_argument(
        '--infer', 
        action='store_true', 
        help="Use Stochastic Weight Perturbation",
    )
    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
