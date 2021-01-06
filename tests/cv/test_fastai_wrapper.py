# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Unit testing the fastai wrapper."""
import unittest
from sotaai.cv import fastai_wrapper


class TestFastaiWrapper(unittest.TestCase):
    """Test fastai's wrapper module."""

    no_check_ds = ["CALTECH_101",   # Not a valid directory.
                   "DOGS",          # Not a valid directory
                   "MNIST_SAMPLE",  # Not a valid directory.
                   ]

    def test_load_dataset_return_type(self):
        """Make sure fastai objects are returned."""
        for task in fastai_wrapper.DATASETS.keys():
            print("Checking task: {}".format(task))
            for ds in fastai_wrapper.DATASETS[task]:
                if ds in self.no_check_ds:
                    continue
                print("Testing {}".format(ds))
                dso = fastai_wrapper.load_dataset(ds)
                # Make sure we are getting a dictionary.
                self.assertEqual(dict, type(dso))
                # Make sure the objects in the dict are fastai.
                for split in dso.keys():
                    print("Split name: {}".format(split))
                    print(type(dso[split]))
                    self.assertTrue('fastai' in str(type(dso[split])))

    def test_model_loading(self):
        """Ensures we can load every model from the fastai module."""
        for model_name in fastai_wrapper.MODELS:
            print("Checking model {}".format(model_name))
            # Models come in a list, and not in a dict...
            model = fastai_wrapper.load_model(model_name)
            print(type(model))
            print("\n")


if __name__ == '__main__':
    unittest.main()
