import unittest
from sphinx.cmd.build import build_main
import os
import shutil


class DocBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        cls.html_path = os.path.join(cls.docs_path, '_build', 'html')
        shutil.rmtree(cls.html_path, ignore_errors=True)

    '''
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.html_path, ignore_errors=True)
    '''

    def test_build_docs(self):
        build_main(argv=['-b', 'html', self.docs_path, self.html_path])


if __name__ == "__main__":
    unittest.main()
