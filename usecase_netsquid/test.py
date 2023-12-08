import os
import unittest
from runpy import run_path
from netsquid.util import logger
from inspect import getfullargspec


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Path to the folder containing this file
        path_to_here = os.path.dirname(os.path.abspath(__file__))

        # Dictionary containing the main functions of the examples with keys being the file-paths
        cls.example_file_paths = []
        for root, folders, files in os.walk(path_to_here):
            for file in files:
                if file.startswith("example_") and file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    cls.example_file_paths.append(file_path)

    def test_examples(self):
        for file_path in self.example_file_paths:
            with self.subTest(file_path=file_path):
                var_dict = run_path(file_path)
                try:
                    example_main = var_dict['main']
                except KeyError:
                    logger.warning("Python file {} does not contain a function called main.".format(file_path))
                else:
                    args_of_main = getfullargspec(example_main).args
                    if "suppress_output" in args_of_main:
                        success = example_main(suppress_output=True)
                    else:
                        logger.warning("main function of Python file {} does not take an argument 'suppress_output."
                                       .format(file_path))
                        continue
                    if success is not None:
                        self.assertTrue(success)


if __name__ == '__main__':
    unittest.main()
