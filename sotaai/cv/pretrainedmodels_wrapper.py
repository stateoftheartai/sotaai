'''
Author: Liubove Orlov Savko

Many thanks to the great work of pretrainedmodels library by Cadene
'''

SOURCE_METADATA = {
    'name': 'pretrainedmodels',
    'original_name': 'pretrained-models',
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


def load_model(name: str):
  return {'name': name, 'source': 'pretrainedmodels'}


def load_dataset(name: str):
  return {'train': {'name': name, 'source': 'pretrainedmodels'}}
