import unittest

# noinspection PyUnresolvedReferences
import propnet.symbols
from propnet.core.registry import Registry


class SymbolsTest(unittest.TestCase):
    def test_property_formatting(self):
        """
        Goes through the Quantity .yaml files and ensures the definitions are complete.
        """
        for st in Registry("symbols").values():
            self.assertTrue(st.name is not None and st.name.isidentifier())
            self.assertTrue(
                st.category is not None
                and st.category in ('property', 'condition', 'object'))
            self.assertTrue(
                st.display_names is not None
                and isinstance(st.display_names, list)
                and len(st.display_names) != 0)
            self.assertTrue(
                st.display_symbols is not None
                and isinstance(st.display_symbols, list)
                and len(st.display_symbols) != 0, st.name)
            self.assertTrue(
                st.comment is not None and isinstance(st.comment, str))
            if st.category != 'object':
                self.assertIsNotNone(
                    st.units,
                    "The property/condition symbol {} is missing units.".
                    format(st.name))

    def test_all_properties(self):
        self.assertEqual(
            str(Registry("symbols")['density'].units),
            'gram / centimeter ** 3')


if __name__ == "__main__":
    unittest.main()
