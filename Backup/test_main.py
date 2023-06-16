import numpy as np
import torch
import os
from torchvision import transforms
import torchvision
import PIL
from PIL import Image
from datasets import Caltech256Images
from torch.utils.data import DataLoader

dataset = Caltech256Images()

train_size = int(len(dataset) * 0.75)                   # training set
test_size = int(len(dataset) * 0.03374)                 # hold-out set for testing
val_size = len(dataset) - (train_size + test_size)      # validation set used during training

train_indices = np.arange(train_size)
val_indices = np.arange(train_size, train_size + val_size)
test_indices = np.arange(train_size + val_size, len(dataset))

train_set = torch.utils.data.Subset(dataset, train_indices)
val_set = torch.utils.data.Subset(dataset, val_indices)
test_set = torch.utils.data.Subset(dataset, test_indices)

train_loader = DataLoader(train_set, batch_size=32, shuffle=False)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print(len(train_set), len(val_set), len(test_set))

test = [x[0] for x in iter(train_loader).next()]

trans = transforms.ToPILImage()
trans(test[0]).show()

# loader = DataLoader(dataset, batch_size=2)
# for idx, batch in enumerate(loader):
#    print(batch[0].shape, batch[1])
#    if idx >= 5:
#        break
