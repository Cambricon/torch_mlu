import re
import argparse
import os

parser = argparse.ArgumentParser(description="Script for update API support list.")

parser.add_argument(
    "--old",
    type=str,
    default="./source/API_support_list/API_support_list.rst",
    help="path to original api support list",
)
parser.add_argument(
    "--new",
    type=str,
    default="./source/API_support_list/API_support_list.rst",
    help="path to the new api support list dumped from autogen_pytorch_apis.py",
)


def replace_lines(file1, file2):
    # Read contents of both files
    with open(file1, "r") as f1, open(file2, "r") as f2:
        content1 = f1.readlines()
        content2 = f2.readlines()

    # Regular expression pattern to match lines starting with "* - torch.xxxx"
    pattern = r"\* \- torch\..+"

    # Iterate through content of the first file
    for i, line in enumerate(content1):
        match = re.findall(pattern, line)
        if match:
            # Get the corresponding section name
            section_name = match[0]
            # Find the matching section in the second file
            for j, line2 in enumerate(content2):
                if section_name == line2.strip():
                    if (
                        "torch.cuda." in section_name
                        or "torch.backends.cuda" in section_name
                        or "torch.backends.cudnn" in section_name
                    ):
                        content1[i + 1] = content2[j + 1]
                        content1[i + 2] = content2[j + 2]
                        content1[i + 3] = content2[j + 3]
                        break
                    else:
                        # Replace the next two lines in the first file with lines from the second file
                        content1[i + 1] = content2[j + 1]
                        content1[i + 2] = content2[j + 2]
                        break
                else:
                    content1[i + 1] = "      - 否\n"

    # Write the modified content back to the first file
    with open(file1, "w") as f1:
        f1.writelines(content1)


# Parse the arguments
args = parser.parse_args()

# New list dumped from python torch_mlu/tools/autogen_pytorch_apis.py 2.3
file1_path = args.new

# original api support list
file2_path = args.old

# Call the function to replace lines
replace_lines(file1_path, file2_path)

print("Lines replaced successfully!")
