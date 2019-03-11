import unittest

from propnet.core.registry import Registry


class RegistryTest(unittest.TestCase):
    def test_basic_registry(self):
        test_reg = Registry("test")
        test_reg2 = Registry("test")
        test_reg3 = Registry("test2")

        self.assertIsInstance(test_reg, dict)
        self.assertTrue(test_reg is test_reg2)
        self.assertTrue(test_reg is not test_reg3)


if __name__ == "__main__":
    unittest.main()
