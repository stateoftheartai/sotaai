# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Main CV module to standarice torch datasets with torch models.'''
import torch.nn as nn
# import numpy as np
import torchvision.transforms as transforms
from PIL import Image


def qmnist_to_alexnet(cv_dataset, cv_model):
  cv_model.raw.features[0] = nn.Conv2d(3,
                                       64,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(1, 1))

  # set output model according with number of dataset labels
  cv_model.raw.classifier[4] = nn.Linear(4096, 1024)
  cv_model.raw.classifier[6] = nn.Linear(1024, 8)

  standarized_model = cv_model.raw

  standarized_dataset = []

  for datapoint in cv_dataset:
    preprocess = transforms.Compose([transforms.ToTensor()])

    image = Image.fromarray(datapoint['image'])

    input_tensor = preprocess(image)
    standarized_element = input_tensor.unsqueeze(0)
    standarized_element = standarized_element.expand(1, 3, 28, 28)
    standarized_dataset.append({
        'image': standarized_element,
        'label': datapoint['label']
    })

  return iter(standarized_dataset), standarized_model
