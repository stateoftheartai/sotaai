# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2021.
'''Unit testing the Torch wrapper.

This is a temporal file to test Torch model_to_dataset for Segmentation task.
This code was not placed in the original test_torch_wrapper.py file to avoid
conflicts temporary. It will be moved there when finished.
'''

# TODO(Hugo)
# Move this code to original test_torch_wrapper.py file when finished.

from sotaai.cv import load_dataset, load_model, utils
import unittest
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as Transforms


class TestTorchWrapperTmp(unittest.TestCase):
  '''Test the wrapped Tortch module.

  Temporal class, it will only test model_to_dataset for Segmentation tasks and
  then moved to the original TestTorchWrapper
  '''

  # This is a temporal method to work on a Segmentation example and being able
  # to estimate for the AA of this task
  def test_segmentation_example(self):

    # All pytorch pre-trained models expect:
    # - (N, 3, H, W), where N is the batch size
    # - N is the batch size
    # - H and W are expected to be at least 224
    # - Pixel values must be in range [0,1] and normilized with mean [0.485,
    # 0.456, 0.406] and std [0.229, 0.224, 0.225]

    dataset_name = 'lost_and_found'
    model_name = 'fcn_resnet101'

    dataset_splits = load_dataset(dataset_name)
    split_name = next(iter(dataset_splits.keys()))
    cv_dataset = dataset_splits[split_name]

    print('Pixel classes', cv_dataset.pixel_classes)

    cv_model = load_model(model_name)
    raw_model = cv_model.raw

    raw_model.classifier[-1] = torch.nn.Conv2d(512,
                                               len(cv_dataset.pixel_classes), 1)

    transform = Transforms.Compose([
        Transforms.ToPILImage(),
        Transforms.Resize(256),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n = 5
    images = []
    numpy_images = []
    for i, item in enumerate(cv_dataset):
      numpy_images.append(item['image'])
      images.append(transform(item['image']))

      if i == n - 1:
        break

    batch = torch.stack(images, dim=0)
    print('input', batch.shape)

    output = raw_model(batch)['out']
    print('output', output.shape)

    figure = plt.figure()

    for i, prediction in enumerate(output):

      mask = torch.argmax(prediction.squeeze(), dim=0).detach().numpy()
      print('prediction', prediction.shape, mask.shape, np.unique(mask))
      segmentation_image = utils.create_segmentation_image(
          mask, len(cv_dataset.pixel_classes))

      figure.add_subplot(n, 2, 2 * i + 1).imshow(numpy_images[i])
      figure.add_subplot(n, 2, 2 * i + 2).imshow(segmentation_image)

    plt.show()


if __name__ == '__main__':
  unittest.main()
