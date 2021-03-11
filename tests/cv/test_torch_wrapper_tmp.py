# -*- coding: utf-8 -*-
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

    dataset_name = 'lost_and_found'
    model_name = 'fcn_resnet101'

    dataset_splits = load_dataset(dataset_name)
    split_name = next(iter(dataset_splits.keys()))
    cv_dataset = dataset_splits[split_name]

    cv_model = load_model(model_name)

    print('\nModel ', cv_model.name)
    print(' Input: ', tuple(cv_model.original_input_shape))
    print(' Output: ', cv_model.original_output_shape)
    print('Dataset: ', cv_dataset.name)
    print(' Shape:   ', cv_dataset.shape)
    print(' Pixel Classes: ', len(cv_dataset.pixel_classes))

    # BEGIN model_to_dataset example logic

    # All pytorch pre-trained models expect:
    # - (N, 3, H, W), where N is the batch size
    # - N is the batch size
    # - H and W are expected to be at least 224
    # - Pixel values must be in range [0,1] and normilized with mean [0.485,
    # 0.456, 0.406] and std [0.229, 0.224, 0.225]

    print('\nAdjusting...')

    raw_model = cv_model.raw

    # TODO(Hugo)
    # Not sure if this is the proper way to change the model output
    raw_model.classifier[-1] = torch.nn.Conv2d(512,
                                               len(cv_dataset.pixel_classes), 1)

    cv_model.update_raw_model(raw_model)

    transform = Transforms.Compose([
        Transforms.ToPILImage(),
        Transforms.Resize(256),
        Transforms.CenterCrop(224),
        Transforms.ToTensor(),
        Transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    n = 3
    images = []
    numpy_images = []
    for i, item in enumerate(cv_dataset):
      numpy_images.append(item['image'])
      images.append(transform(item['image']))

      if i == n - 1:
        break

    batch = torch.stack(images, dim=0)

    # END model_to_dataset example logic

    print('\nModel ', cv_model.name)
    print(' Input: ', tuple(cv_model.original_input_shape))
    print(' Output: ', cv_model.original_output_shape)
    print('Dataset: ', cv_dataset.name)
    print(' Shape:   ', tuple(batch.shape))
    print(' Pixel Classes: ', len(cv_dataset.pixel_classes))

    figure = plt.figure()

    print('\nTesting predictions...')

    output = raw_model(batch)['out']

    print('Sample output shape {}'.format(output.shape))

    for i, prediction in enumerate(output):

      mask = torch.argmax(prediction.squeeze(), dim=0).detach().numpy()

      print('Prediction {} {} {}'.format(i, prediction.shape, mask.shape))

      segmentation_image = utils.create_segmentation_image(
          mask, len(cv_dataset.pixel_classes))

      figure.add_subplot(n, 2, 2 * i + 1).imshow(numpy_images[i])
      figure.add_subplot(n, 2, 2 * i + 2).imshow(segmentation_image)

    plt.show()


if __name__ == '__main__':
  unittest.main()
