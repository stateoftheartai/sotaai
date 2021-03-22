'''
Author: Liubove Orlov Savko

Many thanks to the great work of pretrainedmodels library by Cadene
'''

import pretrainedmodels
from torch import nn

SOURCE_METADATA = {
    'name': 'pretrained-models',
    'original_name': 'pretrainedmodels',
    'url': 'https://github.com/cadene/pretrained-models.pytorch'
}

DATASETS = {}

MODELS = {
    'classification': [
        'fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d',
        'inceptionv4', 'inceptionresnetv2', 'nasnetamobile', 'nasnetalarge',
        'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn131', 'dpn107', 'xception',
        'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
        'se_resnext50_32x4d', 'se_resnext101_32x4d', 'cafferesnet101',
        'pnasnet5large', 'polynet'
    ]
}


def load_model(model_name, classes=1000, pretrained=False):
  '''
    Input:
        model_name: string, one of MODELS
        classes: number of categories to classify
        pretrain: bool
    Output:
        A model of type pretrained.models, which inherits torch.nn.Module
    '''
  # set weights
  if pretrained:
    weights = 'imagenet'
  else:
    weights = None

  # We call the pretrained.models.model_name
  trainer = getattr(pretrainedmodels, model_name)

  return trainer(num_classes=classes, pretrained=weights)


def predict(model, dataset):
  '''
    Input:
        model of type pretrained.models
        dataset: torch.Tensor of shape (N,C,H,W), where
            N - nbr of images
            C - nbr of channels
            H - height of image
            W - width of image
    '''

  result = model(dataset)
  return result


def adapt_last_layer(model, classes):
  '''
    Input:
        model: type pretrained.models
        classes: int, the nbr of classes to output
    '''

  # Make a deep copy of the model so that we don't rewrite the original model
  net = model

  # Get the characteristics of the last layer to replicate for the new layer
  ll = net.last_linear

  # Create new layer to use as last layer in the model
  if isinstance(ll, nn.Linear):
    bias = bool(ll.bias)

    new_layer = nn.Linear(in_features=ll.in_features,
                          out_features=classes,
                          bias=bias)

  bool1 = isinstance(ll, nn.Conv1d)
  bool2 = isinstance(ll, nn.Conv2d)
  bool3 = isinstance(ll, nn.Conv3d)

  new_layer = None
  if bool1 or bool2 or bool3:
    l_type = str(type(ll)).split('.')[-1][:6]
    conv = getattr(nn, l_type)

    bias = bool(ll.bias)

    new_layer = conv(in_channels=ll.in_channels,
                     out_channels=classes,
                     kernel_size=ll.kernel_size,
                     stride=ll.stride,
                     padding=ll.padding,
                     dilation=ll.dilation,
                     groups=ll.groups,
                     bias=bias,
                     padding_mode=ll.padding_mode)
  net.last_linear = new_layer
  return net
