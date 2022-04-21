import os
import WideResNet
import pytorch_lightning as pl
import torchvision
from WideResNet import *
from cr_pl import *
from pytorch_lightning.callbacks import StochasticWeightAveraging, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.strategies.ddp import DDPStrategy

SEED = 2022
seed_everything(SEED)

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
    cr_pl = CR_pl(hparams, backbone)
    
    # ------------------------
    # 2 DEFINE CALLBACKS
    # ------------------------
    bar = TQDMProgressBar(refresh_rate=1, process_position=0)
    #swa_callback = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=None, annealing_epochs=10, 
        #annealing_strategy='cos', avg_fn=None, device=None)
    checkpoint_callback = ModelCheckpoint(
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
    trainer = pl.Trainer(
        progress_bar_refresh_rate=1,
        gpus=hparams.gpus,
        precision=hparams.precision,
        num_nodes=hparams.num_nodes,
        max_epochs=hparams.max_epochs,
        #strategy=None,
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback, bar]
        )

    # ------------------------
    # 4 START TRAINING
    # ------------------------
    trainer.fit(cr_pl)

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

    # each LightningModule defines arguments relevant to it
    parser = CR_pl.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()
    
    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
