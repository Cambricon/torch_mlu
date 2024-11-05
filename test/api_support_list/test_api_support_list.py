import unittest
import os
import subprocess
import yaml
from pathlib import Path


class APISupportListTest(unittest.TestCase):
    def test_lint(self):
        torch_mlu_path = Path(__file__).resolve().parent.parent.parent
        torch_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/torch_api.yaml"
        )
        torchvision_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/torchvision_api.yaml"
        ) 
        custom_ops_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/mlu_custom_operators.yaml"
        )
        torch_api_lint_result = subprocess.run(
            ["yamllint", torch_api_path], capture_output=True, text=True
        )
        self.assertEqual(torch_api_lint_result.stdout, "")
        torchvision_api_lint_result = subprocess.run(
            ["yamllint", torchvision_api_path], capture_output=True, text=True
        )
        self.assertEqual(torchvision_api_lint_result.stdout, "")
        custom_ops_lint_result = subprocess.run(
            ["yamllint", custom_ops_api_path], capture_output=True, text=True
        )
        self.assertEqual(custom_ops_lint_result.stdout, "")

    def yaml_check(self, path):
        with open(path, "r", encoding="utf-8") as api_file:
            try:
                api_data = yaml.safe_load(api_file)
            except yaml.YAMLError as e:
                raise RuntimeError("Failed loading api yaml file") from e
        for i, module in enumerate(api_data):
            module_name = list(module.keys())[0]
            for j, api in enumerate(module[module_name]):
                api_name = list(api.keys())[0]
                for k in api:
                    # use list to ensure object order in single api object
                    api_list = list(api.items())
                    if k not in [
                        "api",
                        "supported",
                        "corresponding MLU api",
                        "limitations",
                    ]:
                        raise RuntimeError(f"Found unkown element {k}: {api[k]}")
                    if "api" not in api or "supported" not in api:
                        raise RuntimeError(
                            f"Structure of yaml was broken, missed api or supported object(s)"
                        )
                    if k == "api" and not isinstance(api[k], str):
                        raise RuntimeError(
                            f"{api[k]} was parsed as a {type(api[k])}, a string value is required for 'api' element"
                        )
                    if k == "corresponding MLU api" and not isinstance(api[k], str):
                        raise RuntimeError(
                            f"{api[k]} was parsed as a {type(api[k])}, a string value is required for 'corresponding MLU api' element"
                        )
                    if k == "supported" and not isinstance(api[k], bool):
                        raise RuntimeError(
                            f"{api[k]} was parsed as a {type(api[k])}, a boolean value is required for 'supported' elemet"
                        )
                    if k == "limitations" and not isinstance(api[k], str):
                        raise RuntimeError(
                            f"{api[k]} was parsed as a {type(api[k])}, a string value is required for 'limitation' element"
                        )

    # For detecting YAML formatting issues that yamllint might miss, such as missing objects/ wrong objects
    def test_api_support_list(self):
        torch_mlu_path = Path(__file__).resolve().parent.parent.parent
        torch_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/torch_api.yaml"
        )
        torchvision_api_path = (
            str(torch_mlu_path)
            + "/tools/autogen_torch_api/api_support_lists/torchvision_api.yaml"
        ) 
        try:
            self.yaml_check(torch_api_path)
        except Exception as e:
            self.fail(f"Check of torch_api.yaml failed with an exception: {e}")
        try:
            self.yaml_check(torchvision_api_path)
        except Exception as e:
            self.fail(f"Check of torchvision_api.yaml failed with an exception: {e}")


if __name__ == "__main__":
    unittest.main()
