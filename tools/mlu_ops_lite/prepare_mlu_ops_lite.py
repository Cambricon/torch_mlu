import os
import json
import subprocess
import sys
import yaml


current_dir = os.path.dirname(os.path.abspath(__file__))
torch_mlu_dir = os.path.join(current_dir, "../../")


def prepare_src():
    try:
        subprocess.run(
            ["bash", f"{current_dir}/mlu_ops_lite.sh"],
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(e)
        sys.exit(1)


op_info = {}
property_file = os.path.join(torch_mlu_dir, "scripts/release/build.property")
with open(property_file, "r") as f:
    json_dict = json.load(f)
    version = json_dict["src_requires"]["mluops-lite"][1]
# current version
op_info["version"] = version

# current operator names
with open(os.path.join(current_dir, "mlu_ops_lite.yaml"), "r") as f:
    operator_names = yaml.safe_load(f)
op_info["operator_names"] = operator_names

# exsisting json
json_file = os.path.join(
    torch_mlu_dir, "torch_mlu/csrc/aten/operators/bang/mlu_ops_lite/mlu_ops_lite.json"
)
# check whether the json file exists
if not os.path.exists(json_file):
    prepare_src()
else:
    with open(json_file, "r") as f:
        existing_content = json.load(f)

    # if version in build.property is different from the existing version, prepare the source code
    existing_version = existing_content["version"]
    if version != existing_version:
        prepare_src()
    else:
        if operator_names != existing_content["operator_names"]:
            prepare_src()

# dump current info to json file
with open(json_file, "w") as f:
    json.dump(op_info, f, indent=2)
