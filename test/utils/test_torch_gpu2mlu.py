import unittest
import tempfile
import os
import sys
import shutil

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import shell

tool_path = os.path.join(cur_dir + "/../../tools/torch_gpu2mlu/", "torch_gpu2mlu.py")


class TorchGPU2MLUScriptTest(unittest.TestCase):
    def test_normal_python_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "gpu_script.py"), "w") as f:
                f.write("import torch\n")
            try:
                return_code = shell(
                    [sys.executable, f"{tool_path}", "-i", f"{temp_dir}"], cwd="/"
                )
                assert return_code == 0, "execute torch_gpu2mlu.py failed!"
                with open(
                    os.path.join(temp_dir + "_mlu", "gpu_script.py"), "r"
                ) as f_out:
                    output = f_out.read()
                    self.assertEqual(output, "import torch\nimport torch_mlu\n")
            finally:
                if os.path.exists(temp_dir + "_mlu"):
                    shutil.rmtree(temp_dir + "_mlu")

    def test_gbk_encoding_python_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(
                os.path.join(temp_dir, "gpu_script.py"), "w", encoding="GBK"
            ) as f:
                f.write("import torch\na='中文编码测试'\n")
            try:
                return_code = shell(
                    [sys.executable, f"{tool_path}", "-i", f"{temp_dir}"], cwd="/"
                )
                assert return_code == 0, "execute torch_gpu2mlu.py failed!"
                with open(
                    os.path.join(temp_dir + "_mlu", "gpu_script.py"),
                    "r",
                    encoding="GBK",
                ) as f_out:
                    output = f_out.read()
                    self.assertEqual(
                        output, "import torch\nimport torch_mlu\na='中文编码测试'\n"
                    )
            finally:
                if os.path.exists(temp_dir + "_mlu"):
                    shutil.rmtree(temp_dir + "_mlu")

    def test_declared_gbk_encoding_python_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(
                os.path.join(temp_dir, "gpu_script.py"), "w", encoding="GBK"
            ) as f:
                f.write("# encoding=gbk\nimport torch\na='中文编码测试'\n")
            try:
                return_code = shell(
                    [sys.executable, f"{tool_path}", "-i", f"{temp_dir}"], cwd="/"
                )
                assert return_code == 0, "execute torch_gpu2mlu.py failed!"
                with open(
                    os.path.join(temp_dir + "_mlu", "gpu_script.py"),
                    "r",
                    encoding="GBK",
                ) as f_out:
                    output = f_out.read()
                    self.assertEqual(
                        output,
                        "# encoding=gbk\nimport torch\nimport torch_mlu\na='中文编码测试'\n",
                    )
            finally:
                if os.path.exists(temp_dir + "_mlu"):
                    shutil.rmtree(temp_dir + "_mlu")

    def test_declared_wrong_encoding_python_script(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(
                os.path.join(temp_dir, "gpu_script.py"), "w", encoding="GBK"
            ) as f:
                f.write("# encoding=ascii\nimport torch\na='中文编码测试'\n")
            try:
                return_code = shell(
                    [sys.executable, f"{tool_path}", "-i", f"{temp_dir}"], cwd="/"
                )
                assert return_code == 0, "execute torch_gpu2mlu.py failed!"
                with open(
                    os.path.join(temp_dir + "_mlu", "gpu_script.py"),
                    "r",
                    encoding="GBK",
                ) as f_out:
                    output = f_out.read()
                    self.assertEqual(
                        output,
                        "# encoding=ascii\nimport torch\nimport torch_mlu\na='中文编码测试'\n",
                    )
            finally:
                if os.path.exists(temp_dir + "_mlu"):
                    shutil.rmtree(temp_dir + "_mlu")


if __name__ == "__main__":
    unittest.main()
