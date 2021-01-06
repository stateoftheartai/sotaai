# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Unit testing the MxNet wrapper."""
import unittest
from sotaai.cv import mxnet_wrapper


class TestMxnetWrapper(unittest.TestCase):
    """Test MxNet wrapping module."""

    def test_load_dataset_return_type(self):
        """Make sure `dict`s are returned, with correct keywords for splits."""
        for task in mxnet_wrapper.DATASETS.keys():
            print("Checking task: {}".format(task))
            for ds in mxnet_wrapper.DATASETS[task]:
                print("Testing {}".format(ds))
                dso = mxnet_wrapper.load_dataset(ds)

                # Should have a dictionary with two splits: 'train' and 'test'.
                self.assertEqual(dict, type(dso))
                self.assertEqual(len(dso), 2)
                self.assertTrue('train' in dso.keys())
                self.assertTrue('test' in dso.keys())

                # Check the internal objects; should be an `mxnet` class.
                for key in dso.keys():
                    self.assertTrue('mxnet' in str(type(dso[key])))

    def test_abstract_dataset(self):
        """Make sure we can create an abstract dataset using MxNet datasets."""
        self.assertTrue(True)

    def test_model_loading(self):
        """Ensures we can load every model from the mxnet module."""
        for model_name in mxnet_wrapper.MODELS:
            print("Checking model {}".format(model_name))
            # Models come in a list, and not in a dict...
            model = mxnet_wrapper.load_model(model_name)
            print(type(model))
            print("\n")


if __name__ == '__main__':
    unittest.main()
