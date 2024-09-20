import re
import argparse
import os
import yaml
import io

parser = argparse.ArgumentParser(description="Script for update API support list.")

parser.add_argument(
    "--old",
    type=str,
    default="./api_support_list.yaml",
    help="path to original api support list",
)
parser.add_argument(
    "--new",
    type=str,
    default="./api_support_list.yaml",
    help="path to the new api support list dumped from autogen_pytorch_apis.py",
)


def replace_lines(file1, file2):
    # Read contents of both files
    with open(file1, "r", encoding="utf-8") as f1, open(
        file2, "r", encoding="utf-8"
    ) as f2:
        content1 = yaml.safe_load(f1)
        content2 = yaml.safe_load(f2)

    for i, new_module in enumerate(content1):
        module_name = list(new_module.keys())[0]
        for old_module in content2:
            old_module_name = list(old_module.keys())[0]
            if module_name == old_module_name:
                if module_name == "Torch Environment Variables":
                    content1[i] = old_module
                else:
                    for j, new_api in enumerate(new_module[module_name]):
                        api_name = new_api["api"]
                        for old_api in old_module[old_module_name]:
                            old_api_name = old_api["api"]
                            if (
                                api_name == old_api_name
                                and old_api["supported"] == True
                            ):
                                content1[i][module_name][j] = old_api

    yaml_string = io.StringIO()
    yaml.dump(
        content1,
        yaml_string,
        width=1000,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    )
    output_yaml = yaml_string.getvalue()
    output_yaml = output_yaml.replace("\\", "")
    yaml_string = io.StringIO(output_yaml)

    # get yaml string
    yaml_content = yaml_string.getvalue()

    # add empty lines
    yaml_lines = yaml_content.splitlines(True)
    yaml_content_with_blank_lines = ""
    previous_line = None
    for i, line in enumerate(yaml_lines):
        yaml_content_with_blank_lines += line
        if i < len(yaml_lines) - 1:
            next_line = yaml_lines[i + 1]
        else:
            next_line = None

        if next_line and next_line.strip().startswith("-"):
            yaml_content_with_blank_lines += "\n"
        previous_line = line

    if yaml_content_with_blank_lines:
        yaml_content_with_blank_lines += "\n"

    with open("./api_support_list/torch_api.yaml", "w", encoding="utf-8") as yaml_file:
        yaml_file.write(yaml_content_with_blank_lines)


# Parse the arguments
args = parser.parse_args()

# New list dumped from python torch_mlu/tools/autogen_pytorch_apis.py 2.3
file1_path = args.new

# original api support list
file2_path = args.old

# Call the function to replace lines
replace_lines(file1_path, file2_path)

print("API Support List updated successfully!")
