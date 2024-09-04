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


class TestKinetoTbPlugin(TestCase):
    def get_executable_command(self):
        executable = [sys.executable]
        return executable

    def get_test_files(self, path):
        pyfiles = [
            filename
            for filename in glob.glob("{}/test*.py".format(path), recursive=True)
        ]
        return pyfiles

    def install_tb_plugin(self, base_dir):
        total_error_info = []
        os.chdir(base_dir)
        cmd = "pip install -e ."
        args = shlex.split(cmd)
        return_code = shell(args)
        gen_err_message(return_code, "Install tb_plugin", total_error_info)

        print("*********** Install tb_plugin: Error Message Summaries **************")
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print("**********************************************************************")

        if total_error_info:
            raise RuntimeError("Install tb_plugin Failed")

    def run_test(self, executable, test_files, test_directory):
        total_error_info = []
        commands = (executable + [argv] for argv in test_files)
        for command in commands:
            return_code = shell(command, test_directory)
            gen_err_message(return_code, command[-1], total_error_info)

        # Print total error message
        print("*********** Test tb_lugin : Error Message Summaries **************")
        for err_message in total_error_info:
            logging.error("\033[31;1m {}\033[0m .".format(err_message))
        print("*****************************************************************")

        if total_error_info:
            raise RuntimeError("Test tb_plugin Failed")

    def test_mlu_tb_plugin(self):
        executable_ = self.get_executable_command()
        tb_plugin_dir = os.path.join(
            cur_dir, "../..", "third_party/kineto_mlu/tb_plugin"
        )
        test_dir = os.path.join(tb_plugin_dir, "test")
        self.install_tb_plugin(tb_plugin_dir)
        pyfiles_ = self.get_test_files(test_dir)
        self.run_test(executable_, pyfiles_, test_dir)


if __name__ == "__main__":
    unittest.main()
