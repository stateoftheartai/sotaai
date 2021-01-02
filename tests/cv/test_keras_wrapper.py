# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.

import unittest
import numpy as np
from sotaai.cv import keras_wrapper

# @todo
# Still pending to add cv/abstractions.py for the migrated datasets/models only
# from sotaai.cv.abstractions import AbstractCvDataset


class TestKerasWrapper(unittest.TestCase):

    def test_load_dataset_return_type(self):

        all_keywords = []
        for task in keras_wrapper.DATASETS.keys():
            print("Checking task: {}".format(task))
            for ds in keras_wrapper.DATASETS[task]:
                print("*********************")
                print("Testing {}".format(ds))
                dso = keras_wrapper.load_dataset(ds)

                # Make sure we are receiving a dictionary.
                self.assertEqual(type(dso), dict)

                # Make sure the dictionary has the correct keys
                print(dso.keys())

                # Make sure internal objects are tuples of type numpy array.
                for key in dso.keys():
                    all_keywords.append(key)
                    # Each dataset split should be a tuple of two elements.
                    self.assertEqual(tuple, type(dso[key]))
                    self.assertEqual(len(dso[key]), 2)

                    # The tuple's elements need to be `numpy.ndarray`s.
                    self.assertEqual(np.ndarray, type(dso[key][0]))
                    self.assertEqual(np.ndarray, type(dso[key][1]))

        print(all_keywords)
        print(set(all_keywords))

    # def test_abstract_dataset(self):
    #     """
    #         Make sure we can create an abstract dataset
    #         using Keras datasets.
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

    def test_model_loading(self):
        """Make sure that we can load every model from the Keras module."""
        for model_name in keras_wrapper.MODELS:
            print("Checking model {}".format(model_name))
            # All models are for classification in Keras.
            model = keras_wrapper.load_model(model_name)
            print(type(model))
            print("\n")


if __name__ == '__main__':
    unittest.main()
