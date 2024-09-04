import sys
import os
import re
import unittest
import logging
import warnings
import torch
import torch_mlu
from torch_mlu.utils.gpu_migration.env_migration import mlu_env_map

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import run_tests, TestCase, testinfo

logging.basicConfig(level=logging.DEBUG)


class TestEnvMigration(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        required_env_list = ["PYTORCH_HOME", "TORCH_MLU_HOME"]
        for var in required_env_list:
            value = os.environ.get(var)
            if not value:
                raise EnvironmentError(f"Please set the {var} before checking.")
            setattr(cls, var.lower(), value)

    def find_env_in_file(self, file_path, patterns):
        env_set = set()
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                for pattern in patterns:
                    env_set.update(pattern.findall(line))
        return env_set

    def find_mlu_env_in_file(self, file_path, patterns):
        env_set = set()
        with open(file_path, "r", encoding="utf-8") as file:
            buffer = ""
            for line in file:
                buffer += line.strip()
                if patterns[2].search(line):  # pattern cross line
                    continue
                env_set.update(patterns[0].findall(buffer))  # pattern single
                matches_multiple = patterns[1].findall(buffer)  # pattern multiple
                for match in matches_multiple:
                    var_list = [
                        var.strip().strip('"').strip() for var in match.split(",")
                    ]
                    env_set.update(var_list)
                env_set.update(patterns[3].findall(line))  # pattern check_env
                buffer = ""
        return env_set

    # Get the latest CUDA environment variables.
    def find_cuda_env(self):
        # PyTorch 2.1.0 lacks CUDA environment variable docs, so check the source code.
        search_folders = [
            os.path.join(self.pytorch_home, "aten", "src", "ATen"),
            os.path.join(self.pytorch_home, "c10"),
            os.path.join(self.pytorch_home, "torch", "cuda"),
        ]

        # Patterns to match getenv and check_env with specific keywords
        cuda_keywords = ["CUBLASLT", "CUBLAS", "CUDA", "CUDNN", "CUSOLVER"]
        patterns = [
            re.compile(r'\bgetenv\(\s*"([A-Za-z0-9_]+)"\s*\)'),
            re.compile(r'\bcheck_env\(\s*"([A-Za-z0-9_]+)"\s*\)'),
            re.compile(r'\benv_flag_set\(\s*"([A-Za-z0-9_]+)"\s*\)'),
        ]

        all_env_set = set()
        for folder in search_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith((".cpp", ".hpp", ".h", ".py")):
                        file_path = os.path.join(root, file)
                        env_set = self.find_env_in_file(file_path, patterns)
                        all_env_set.update(env_set)

        # Filter out environment variables that don't contain any of the keywords
        cuda_env_list = [
            var
            for var in all_env_set
            if any(keyword in var for keyword in cuda_keywords)
        ]

        print(f"Found {len(cuda_env_list)} CUDA environment variables:")
        for cuda_env_var in sorted(cuda_env_list):
            print(cuda_env_var)

        return sorted(cuda_env_list)

    # Get the latest NCCL environment variables.
    def find_nccl_env(self):
        c10d_folder = os.path.join(
            self.pytorch_home, "torch", "csrc", "distributed", "c10d"
        )
        patterns = [
            re.compile(r'"(TORCH_NCCL_[A-Z_]+)"'),
            re.compile(r'"(NCCL_[A-Z_]+)"'),
            re.compile(r"\b(TORCH_NCCL_[A-Z_]+)\b"),
            re.compile(r"\b(NCCL_[A-Z_]+)\b"),
        ]

        all_env_set = set()
        for root, _, files in os.walk(c10d_folder):
            for file in files:
                if file.endswith((".cpp", ".hpp", ".h")):
                    file_path = os.path.join(root, file)
                    env_set = self.find_env_in_file(file_path, patterns)
                    all_env_set.update(env_set)

        torch_nccl_env_list = [
            var for var in all_env_set if var.startswith("TORCH_NCCL_")
        ]
        nccl_env_list = [
            var
            for var in all_env_set
            if var.startswith("NCCL_") and f"TORCH_{var}" in torch_nccl_env_list
        ]

        print(
            f"Found {len(torch_nccl_env_list)} + {len(nccl_env_list)} NCCL environment variables:"
        )
        for torch_nccl_env_var in sorted(torch_nccl_env_list):
            print(torch_nccl_env_var)
        for nccl_env_var in sorted(nccl_env_list):
            print(nccl_env_var)

        return sorted(torch_nccl_env_list) + sorted(nccl_env_list)

    # Get the latest MLU environment variables.
    def find_mlu_env(self):
        search_folders = [
            os.path.join(self.torch_mlu_home, "torch_mlu/csrc/utils"),
            os.path.join(self.torch_mlu_home, "torch_mlu/csrc/framework/core"),
            os.path.join(self.torch_mlu_home, "torch_mlu/csrc/framework/distributed"),
            os.path.join(self.torch_mlu_home, "pytorch_patches"),
        ]
        exclude_file = ["Utils.h"]
        functions = ["getCvarString", "getCvarInt", "getCvarBool"]
        patterns = [
            re.compile(
                r"\b(?:" + "|".join(functions) + r")\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,"
            ),
            re.compile(r"\b(?:" + "|".join(functions) + r")\(\s*{([^}]+)}\s*,"),
            re.compile(r"\b(?:" + "|".join(functions) + r")\(\s*$"),
            re.compile(r'\bcheck_env\(\s*"([A-Za-z0-9_]+)"\s*\)'),
        ]

        all_env_set = set()
        for folder in search_folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if (
                        file.endswith((".cpp", ".hpp", ".h", ".diff"))
                        and file not in exclude_file
                    ):
                        file_path = os.path.join(root, file)
                        env_set = self.find_mlu_env_in_file(file_path, patterns)
                        all_env_set.update(env_set)

        # Define the keywords for filtering
        mlu_keywords = ["TORCH_CNCL", "MLU", "CAMBRICON", "CNMATMUL"]
        mlu_env_list = [
            var
            for var in all_env_set
            if any(keyword in var for keyword in mlu_keywords)
        ]

        print(f"Found {len(mlu_env_list)} MLU environment variables:")
        for mlu_env_var in mlu_env_list:
            print(mlu_env_var)

        return mlu_env_list

    def replace(self, mlu_env):
        replaced_env_list = []
        if "CNCL" in mlu_env:
            torch_nccl_env = mlu_env.replace("CNCL", "NCCL")
            nccl_env = mlu_env.replace("TORCH_CNCL", "NCCL")
            replaced_env_list = [torch_nccl_env, nccl_env]
        elif "MLU" in mlu_env:
            cuda_env = mlu_env.replace("MLU", "CUDA")
            replaced_env_list = [cuda_env]
        elif "CAMBRICON" in mlu_env:
            nvidia_env = mlu_env.replace("CAMBRICON", "NVIDIA")
            replaced_env_list = [nvidia_env]
        elif "CNMATMUL" in mlu_env:
            cublas_env = mlu_env.replace("CNMATMUL", "CUBLAS")
            replaced_env_list = [cublas_env]
        else:
            warnings.warn(f"No corresponding replacement rules found.")
        return replaced_env_list

    def gen_new_map(self, new_env, del_env, mlu_new_env, mlu_del_env):
        # Update Pytorch CUDA environment variables
        for key in del_env:
            del mlu_env_map[key]
        for key in new_env:
            mlu_env_map[key] = ""
        # Update Pytorch MLU environment variables
        for key, value in mlu_env_map.items():
            if value in mlu_del_env:
                mlu_env_map[key] = ""
        for mlu_value in mlu_new_env:
            cuda_env_list = self.replace(mlu_value)
            for cuda_key in cuda_env_list:
                if cuda_key in mlu_env_map:
                    mlu_env_map[cuda_key] = mlu_value
        return mlu_env_map

    def get_diff(self, latest_list, current_list, device):
        diff_latest_current = list(set(latest_list) - set(current_list))
        diff_current_latest = list(set(current_list) - set(latest_list))
        if not diff_latest_current and not diff_current_latest:
            print(f"PyTorch {device} environment variables do not need to be updated.")
        else:
            if diff_latest_current:
                print(
                    f"The newly added PyTorch {device} environment variables are as follows: "
                    f"{diff_latest_current}"
                )
            if diff_cur_lat:
                print(
                    f"The following PyTorch {device} environment variables have been deprecated: "
                    f"{diff_current_latest}"
                )

        return diff_latest_current, diff_current_latest

    @testinfo()
    def test_latest_env(self):
        # Check the latest environment variables in the current version
        cuda_env_list = self.find_cuda_env()
        nccl_env_list = self.find_nccl_env()
        latest_env_list = cuda_env_list + nccl_env_list
        latest_mlu_env_list = self.find_mlu_env()
        # Get the current environment variables in mlu_env_map
        current_env_list = [key for key in mlu_env_map.keys()]
        current_mlu_env_list = [value for value in mlu_env_map.values() if value]
        # Get the newly added and deleted environment variables
        new_env, del_env = self.get_diff(latest_env_list, current_env_list, "CUDA")
        mlu_new_env, mlu_del_env = self.get_diff(
            latest_mlu_env_list, current_mlu_env_list, "MLU"
        )
        # Generate a new mlu_env_map if the current version has updates
        new_mlu_env_map = {}
        if any([new_env, del_env, mlu_new_env, mlu_del_env]):
            new_mlu_env_map = self.gen_new_map(
                new_env, del_env, mlu_new_env, mlu_del_env
            )
            print(f"The new mlu_env_map is as follows: " f"{new_mlu_env_map}")
        self.assertFalse(new_mlu_env_map)


if __name__ == "__main__":
    run_tests()
