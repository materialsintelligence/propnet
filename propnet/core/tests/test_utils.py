import unittest

from propnet.core.utils import uuid


class UuidTest(unittest.TestCase):

    def test_uuid(self):

        my_uuid = uuid("test_name")
        self.assertEqual(str(my_uuid), "2b49c27c-ffe7-593b-b08c-69c88155b771")
