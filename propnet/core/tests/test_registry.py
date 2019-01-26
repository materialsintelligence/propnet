import unittest

from propnet.core.registry import RegistryMeta


class registerable(metaclass=RegistryMeta):
    fail_on_duplicate = False

    def __init__(self, name=None):
        self.name = name


class registerable_no_duplicates(registerable):
    fail_on_duplicate = True

    def __init__(self, name):
        self.name = name


class registerable_multi_inheritance(type, registerable):
    pass


class RegistryTest(unittest.TestCase):
    def test_basic_registry(self):
        for i in range(10):
            name = str(i)
            registerable(name)

        for i in range(10):
            # Ensure registry has the names
            self.assertIn(str(i), registerable.all_instances.keys())
            # Ensure registry is accessible via index operation
            self.assertTrue(isinstance(registerable[str(i)], registerable))

        with self.assertRaises(ValueError):
            registerable()

    def test_registry_inheritance(self):

        with self.assertRaises(ValueError):
            registerable_no_duplicates("1")
            registerable_no_duplicates("1")

        self.assertEqual(registerable_no_duplicates.all_instances, registerable.all_instances)


if __name__ == "__main__":
    unittest.main()
