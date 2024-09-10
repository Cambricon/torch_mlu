import subprocess
import shlex
import sys
import os
import logging
import glob
import unittest

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import shell, gen_err_message, TestCase

logging.basicConfig(level=logging.DEBUG)


class TestMluExtension(TestCase):
    def get_executable_command(self):
        executable = [sys.executable]
        return executable

    def get_test_files(self, path):
        pyfiles = [
            filename
            for filename in glob.glob("{}/**/test*.py".format(path), recursive=True)
        ]
        return pyfiles

    def build_extension(self, extension_dir):
        # from test file to mlu_extension's directory
        total_error_info = []
        os.chdir(extension_dir)
        command = "python setup.py install"
        args = shlex.split(command)
        return_code = shell(args, extension_dir)
        gen_err_message(return_code, "Mlu_extension", total_error_info)

        print(
            "*********** MLUExtension Mlu_custom_ext Build : Error Message Summaries **************"
        )
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print(
            "*******************************************************************************"
        )

        if total_error_info:
            raise RuntimeError("MLUExtension Mlu_custom_ex Build Failed")

    def run_test(self, executable, test_files, test_directory):
        total_error_info = []
        commands = (executable + [argv] for argv in test_files)
        for command in commands:
            return_code = shell(command, test_directory)
            gen_err_message(return_code, command[-1], total_error_info)

        # Print total error message
        print(
            "*********** MLUExtension test_sigmoid : Error Message Summaries **************"
        )
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print(
            "*******************************************************************************"
        )

        if total_error_info:
            raise RuntimeError("MLUExtension test_sigmoid case Failed")

    def test_mlu_extension(self):
        if not os.getenv("MLU_VISIBLE_DEVICES"):
            os.environ["MLU_VISIBLE_DEVICES"] = "0"
        executable_ = self.get_executable_command()
        base_dir = os.path.join(cur_dir, "../..", "examples", "mlu_extension")
        self.build_extension(base_dir)
        pyfiles_ = self.get_test_files(base_dir)
        self.run_test(executable_, pyfiles_, cur_dir)


if __name__ == "__main__":
    unittest.main()
