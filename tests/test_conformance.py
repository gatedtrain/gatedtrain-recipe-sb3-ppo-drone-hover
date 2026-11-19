"""The adapter `sb3_ppo` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "sb3_ppo"


if __name__ == "__main__":
    unittest.main()
