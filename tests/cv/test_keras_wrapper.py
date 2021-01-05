# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.

"""Unit testing the Keras wrapper."""

import os
import unittest
# import numpy as np
from sotaai.cv import keras_wrapper
# from sotaai.cv.abstractions import AbstractCvDataset
from tensorflow.python.keras.engine.functional import Functional
import inspect


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class TestKerasWrapper(unittest.TestCase):
    """Test the wrapped Keras module."""

    #
    # @author Hugo Ochoa
    # Function temporary commented to avoid testexecution as Github Action
    # Since these tests require dataset to be downloaded
    # @todo check how to better do this in the CI server
    #
    # def test_load_dataset(self):
    #     """
    #       Make sure `dict`s are returned, with correct keywords for splits.
    #     """
    #     all_keywords = []
    #     for task in keras_wrapper.DATASETS.keys():
    #         for dataset_name in keras_wrapper.DATASETS[task]:

    #             dataset = keras_wrapper.load_dataset(dataset_name)

    #             self.assertEqual(type(dataset), dict)

    #             for key in dataset.keys():
    #                 all_keywords.append(key)
    #                 self.assertEqual(tuple, type(dataset[key]))
    #                 self.assertEqual(len(dataset[key]), 2)

    #                 self.assertEqual(np.ndarray, type(dataset[key][0]))
    #                 self.assertEqual(np.ndarray, type(dataset[key][1]))

    def test_load_model(self):
        """Make sure that we can load every model from the Keras module."""
        for model_name in keras_wrapper.MODELS:
            model = keras_wrapper.load_model(model_name)

            self.assertIsInstance(model, Functional)

            self.assertEqual(inspect.ismethod(model.compile), True)
            self.assertEqual(inspect.ismethod(model.fit), True)
            self.assertEqual(inspect.ismethod(model.predict), True)
            self.assertEqual(inspect.ismethod(model.summary), True)
            self.assertEqual(inspect.ismethod(model.save), True)

    #
    # @author Hugo Ochoa
    # @todo Finish abstraction tests
    #
    # def test_abstract_dataset(self):
    #     """
    #       Make sure we can create an abstract dataset using
    #       Keras datasets.
    #     """
    #     # All Keras datasets are for classification tasks.
    #     for task in keras_wrapper.DATASETS.keys():
    #         print("Checking task: {}".format(task))
    #         for ds in keras_wrapper.DATASETS[task]:
    #             dso = keras_wrapper.load_dataset(ds)

    #             # Create one standardized, abstract dataset object per split.
    #             ads = dict()
    #             for key in dso.keys():
    #                 ads[key] = AbstractCvDataset(dso[key], ds, 'image', key,
    #                                              'classification')
    #                 print(ads[key].source)
    #                 print(ads[key].size)
    #                 print(ads[key].shape)
    #             print(ads)


if __name__ == '__main__':
    unittest.main()
