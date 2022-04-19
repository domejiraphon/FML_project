# FML_project
## To install
```
pip install -r requirement.txt
```
## To run with no attack training
```
python main.py --model_path test --restart
```

## To run with PGD training
```
python main.py --model_path attack --attack --restart
``` 

This will create tensorboard file where you can monitor the training schedule. In the tensorboard, you can find thhe graph for adversarial attack score from AutoAttack.

## To use tensorboard
```
tensorboard --logdir=./runs
```