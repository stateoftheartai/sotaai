# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''
Example 1

It shows:
- How to load a model and a dataset
- How to make a model compatible with a dataset
- How to iterate over a dataset and create a batch
- How to execute a model with a batch of data to get predictions
'''
from sotaai.cv import load_dataset, load_model, model_to_dataset
import numpy as np

model = load_model('ResNet152')
dataset = load_dataset('mnist')

dataset_split = dataset['train']

model, dataset = model_to_dataset(model, dataset_split)

batch = []
batch_size = 10

for i, item in enumerate(dataset_split):

  if i == batch_size:
    break

  image_sample = item['image']
  batch.append(image_sample)

batch = np.array(batch)

print('Batch shape', batch.shape)

predictions = model(batch)

print('Predictions shape', predictions.shape)
