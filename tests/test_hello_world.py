# -*- coding: utf-8 -*-
# Author: Hugo Ochoa <hugo@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.

"""Unit testing hello_world function"""

import unittest
from sotaai import hello_world


class TestHelloWorld(unittest.TestCase):

    """Test sotaai.hello_world function"""

    def test_hello_world(self):

        hw = hello_world()

        self.assertEqual(type(hw), str)
        self.assertEqual(hw, 'Hello World from sota.ai')


if __name__ == '__main__':
    unittest.main()
