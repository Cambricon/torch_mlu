# --coding:utf-8--
import json
import os
import shutil
import yaml


def copy_files_from_json(json_data, operator_names, src_path, dst_path):
    if not os.path.exists(src_path):
        raise ValueError(f"Source path {src_path} does not exist.")

    # copy common files
    common_files = json_data.get("common", [])
    for file in common_files:
        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(dst_path, file)
        dst_dir = os.path.dirname(dst_file)
        # not only copy file, but also copy dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_dir)
        else:
            raise ValueError(f"Common file {file} does not exist.")

    # find the specific ops, copy to dst path
    for operator in json_data.get("operators", []):
        if operator["name"] in operator_names:
            header = operator.get("header", [])
            if isinstance(header, str):
                header = [header]
            sources = operator.get("sources", [])
            if isinstance(sources, str):
                sources = [sources]
            for file in header + sources:
                # src can be file or dir
                src = os.path.join(src_path, file)
                if os.path.exists(src):
                    if os.path.isfile(src):
                        shutil.copy(src, dst_path)
                    elif os.path.isdir(src):
                        # copy all files in src dir to dst path
                        for item in os.listdir(src):
                            src_file = os.path.join(src, item)
                            if os.path.isfile(src_file):
                                shutil.copy(src_file, dst_path)
                else:
                    raise ValueError(
                        f"File {file} for operator {operator['name']} does not exist."
                    )


current_dir = os.path.dirname(os.path.abspath(__file__))
mlu_ops_dir = os.path.join(current_dir, "../../third_party/mlu-ops")

with open(os.path.join(current_dir, "mlu_ops_lite.yaml"), "r") as f:
    operator_names = yaml.safe_load(f)

dst_path = os.path.join(
    current_dir, "../../torch_mlu/csrc/aten/operators/bang/mlu_ops_lite/"
)

bangc_kernels_path = os.path.join(
    mlu_ops_dir, "scripts/bangc_kernels_path_config/bangc_kernels_path.json"
)
with open(bangc_kernels_path, "r") as f:
    json_data = json.load(f)
copy_files_from_json(json_data, operator_names, mlu_ops_dir, dst_path)

bangc_kernels_path = os.path.join(
    mlu_ops_dir, "scripts/bangc_kernels_path_config/bangc_kernels_path_extend.json"
)
if os.path.exists(bangc_kernels_path):
    with open(bangc_kernels_path, "r") as f:
        json_data = json.load(f)
    copy_files_from_json(json_data, operator_names, mlu_ops_dir, dst_path)
