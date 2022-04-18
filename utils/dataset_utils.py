import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class CIFAR10(torchvision.datasets.CIFAR10):
  """Returns  """

  def __init__(self, **kwargs):
    super(CIFAR10, self).__init__(**kwargs)
    self.data = torch.from_numpy(self.data).permute([0, 3, 1, 2])
    self.targets = torch.from_numpy(np.array(self.targets))
    
    if torch.max(self.data).item() > 1.0:
      self.data = self.data / 255.0
   
  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, index: int):
    """
    Args:
        index (int): Index
    Returns:
        Dict: (image, target) where target is index of the target class.
    """
    img, labels = self.data[index], self.targets[index]
  
    return {"img": img,
            "labels": labels}

def get_dataset(cfg):
  dataset = {"train": None, "test": None}
  dataloader = dataset.copy()
  for data_type in dataset.keys():
    dataset[data_type] = CIFAR10(
                root=cfg.dataset_path,
                train=True if data_type =="train" else False,
                download=True)
    
    dataloader[data_type] = DataLoader(dataset[data_type], 
                              shuffle=True if data_type =="train" else False,
                              batch_size=cfg.batch_size,
                              num_workers=cfg.num_workers)
  
  return dataset, dataloader
