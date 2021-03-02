# -*- coding: utf-8 -*-
# Author: Tonio Teran
# Copyright: Stateoftheart AI PBC 2021.
'''Hugging Face wrapper module.'''

from typing import Optional, Union, Dict, List

from datasets.features import Features
from datasets.utils import DownloadConfig, GenerateMode, Version
from datasets.splits import Split
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import load_dataset as load_dataset_huggingface

from transformers import AutoModel


def load_model(name: str, *model_args, **kwargs):
  '''Gets a model directly from Hugging Face library.

    Args:
      name: Name/Path of the model to be gotten.
      Model names available at https://huggingface.co/models

    Returns:
      Hugging face model. All models intherits from PreTrainedModel
  '''

  return AutoModel.from_pretrained(name, *model_args, **kwargs)


def load_dataset(path: str,
                 name: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 data_files: Union[Dict, List] = None,
                 split: Optional[Union[str, Split]] = None,
                 cache_dir: Optional[str] = None,
                 features: Optional[Features] = None,
                 download_config: Optional[DownloadConfig] = None,
                 download_mode: Optional[GenerateMode] = None,
                 ignore_verifications: bool = False,
                 keep_in_memory: bool = False,
                 save_infos: bool = False,
                 script_version: Optional[Union[str, Version]] = None,
                 use_auth_token: Optional[Union[bool, str]] = None,
                 **config_kwargs) -> Union[DatasetDict, Dataset]:
  '''Gets a dataset directly from Hugging Face library.

    Args:
        path (``str``):
            path to the dataset processing script with the dataset builder. Can be either:
                - a local path to processing script or the directory containing the script (if the script has the same name as the directory),
                    e.g. ``'./dataset/squad'`` or ``'./dataset/squad/squad.py'``
                - a dataset identifier in the HuggingFace Datasets Hub (list all available datasets and ids with ``datasets.list_datasets()``)
                    e.g. ``'squad'``, ``'glue'`` or ``'openai/webtext'``
        name (Optional ``str``): defining the name of the dataset configuration
        data_files (Optional ``str``): defining the data_files of the dataset configuration
        data_dir (Optional ``str``): defining the data_dir of the dataset configuration
        split (`datasets.Split` or `str`): which split of the data to load.
            If None, will return a `dict` with all splits (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`).
            If given, will return a single Dataset.
            Splits can be combined and specified like in tensorflow-datasets.
        cache_dir (Optional ``str``): directory to read/write data. Defaults to "~/datasets".
        features (Optional ``datasets.Features``): Set the features type to use for this dataset.
        download_config (Optional ``datasets.DownloadConfig``: specific download configuration parameters.
        download_mode (Optional `datasets.GenerateMode`): select the download/generate mode - Default to REUSE_DATASET_IF_EXISTS
        ignore_verifications (bool): Ignore the verifications of the downloaded/processed dataset information (checksums/size/splits/...)
        keep_in_memory (bool, default=False): Whether to copy the data in-memory.
        save_infos (bool): Save the dataset information (checksums/size/splits/...)
        script_version (Optional ``Union[str, datasets.Version]``): Version of the dataset script to load:
            - For canonical datasets in the `huggingface/datasets` library like "squad", the default version of the module is the local version fo the lib.
            You can specify a different version from your local version of the lib (e.g. "master" or "1.2.0") but it might cause compatibility issues.
            - For community provided datasets like "lhoestq/squad" that have their own git repository on the Datasets Hub, the default version "main" corresponds to the "main" branch.
            You can specify a different version that the default "main" by using a commit sha or a git tag of the dataset repository.
        use_auth_token (Optional ``Union[str, bool]``): Optional string or boolean to use as Bearer token for remote files on the Datasets Hub.
            If True, will get token from ~/.huggingface.
        **config_kwargs (Optional ``dict``): keyword arguments to be passed to the ``datasets.BuilderConfig`` and used in the ``datasets.DatasetBuilder``.

    Returns:
      ``datasets.Dataset`` or ``datasets.DatasetDict``
            if `split` is not None: the dataset requested,
            if `split` is None, a ``datasets.DatasetDict`` with each split.
    '''
  return load_dataset_huggingface(path,
                                  name=name,
                                  data_dir=data_dir,
                                  data_files=data_files,
                                  split=split,
                                  cache_dir=cache_dir,
                                  features=features,
                                  download_config=download_config,
                                  download_mode=download_mode,
                                  ignore_verifications=ignore_verifications,
                                  keep_in_memory=keep_in_memory,
                                  save_infos=save_infos,
                                  script_version=script_version,
                                  use_auth_token=use_auth_token,
                                  **config_kwargs)
