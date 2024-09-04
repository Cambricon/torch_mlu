import os
import re
import argparse
import shutil

import torch
import torch_mlu


def find_encoding_declaration(file_path):
    encoding_patterns = [
        "coding[:=]\s*([-\w.]+)"
    ]  # Regular expression to find the encoding
    try:
        with open(file_path, "r", encoding="ascii", errors="ignore") as file:
            first_line = file.readline()
            second_line = file.readline()
            for line in [first_line, second_line]:
                match = re.search(encoding_patterns[0], line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"An error occurred: {e} when get encoding declaration of {file_path}")
        raise
    return None


def detect_encoding(file_path):
    with open(file_path, "rb") as file:  # Open the file in binary mode
        rawdata = file.read()
    import chardet

    result = chardet.detect(rawdata)
    return result["encoding"]


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", "-i", default="./", required=True, type=str, help="input file."
)
parser.add_argument(
    "--strict", "-s", action="store_true", help="strict conversion mode."
)

args = parser.parse_args()

input_path = os.path.abspath(args.input)
mlu_path = input_path + "_mlu"

if not os.path.exists(mlu_path):
    os.mkdir(mlu_path)

mlu_report = os.path.join(mlu_path, "report.md")
report = open(mlu_report, "w+")

report.write("# Cambricon PyTorch Model Migration Report\n")
report.write("## Cambricon PyTorch Changes\n")
report.write("| No. |  File  |  Description  |\n")

num = 0
regex_dict = {
    "cuda": "mlu",
    "nccl": "cncl",
    "torch.backends.cudnn.enabled": "torch.backends.mlufusion.enabled",
    "torch.backends.cuda.matmul.allow_tf32": "torch.backends.mlu.matmul.allow_tf32",
    "torch.backends.cudnn.allow_tf32": "torch.backends.cnnl.allow_tf32",
    "ProfilerActivity.CUDA": "ProfilerActivity.MLU",
    "nvtx": "cnpx",
    "torch.get_autocast_gpu_dtype": "torch.mlu.get_autocast_dtype",
    "torch.set_autocast_gpu_dtype": "torch.mlu.set_autocast_dtype",
    "torch.utils.cpp_extension.load": "torch_mlu.utils.cpp_extension.load",
    "torch.utils.cpp_extension.load_inline": "torch_mlu.utils.cpp_extension.load_inline",
    "CUDAGraph": "MLUGraph",
}
regex_strict_dict = {"CUDA": "MLU", "gpu": "mlu", "GPU": "MLU"}
if args.strict:
    regex_dict.update(regex_strict_dict)

whitelist_regex_dict = {}
for root, dirs, files in os.walk(input_path):
    for file in files:
        try:
            file_path = os.path.join(root, file)
            relative_path = file_path[len(input_path) + 1 :]
            mlu_file_path = mlu_path + file_path[len(input_path) :]
            root_mlu = os.path.dirname(mlu_file_path)
            if not os.path.exists(root_mlu):
                os.makedirs(root_mlu)

            if not file_path.endswith(".py"):
                try:
                    shutil.copy(file_path, mlu_file_path)
                except:
                    print("copy error: ", file_path)

            else:
                # We will firstly use the explicitly specified character encoding info, then consider
                # auto detect as chardet recommended. Ref https://chardet.readthedocs.io/en/latest/faq.html
                f = None
                try:
                    text_encoding = find_encoding_declaration(file_path)
                    f = open(file_path, "r+", encoding=text_encoding)
                    file_cont = f.readlines()
                except:
                    if f:
                        f.close()
                    text_encoding = detect_encoding(file_path)
                    f = open(file_path, "r+", encoding=text_encoding)
                    file_cont = f.readlines()
                mlu_f = open(mlu_file_path, "w+", encoding=text_encoding)
                line = 0
                has_import = False
                for ss in file_cont:
                    line = line + 1
                    if ss.strip() == "import torch" and not has_import:
                        num = num + 1
                        has_import = True
                        mlu_f.write(ss)
                        ss = re.sub("torch", "torch_mlu", ss)
                        mlu_f.write(ss)
                        report.write(
                            "| "
                            + str(num)
                            + " | "
                            + relative_path
                            + ":"
                            + str(line)
                            + ' | add "import torch_mlu" |\n'
                        )
                        continue

                    ori_ss = ss
                    for key in regex_dict.keys():
                        ss = re.sub(key, regex_dict[key], ss)

                    # avoid whitelist APIs are converted.
                    for key in whitelist_regex_dict.keys():
                        ss = re.sub(whitelist_regex_dict[key], key, ss)

                    mlu_f.write(ss)
                    if ori_ss != ss:
                        num = num + 1
                        report.write(
                            "| "
                            + str(num)
                            + " | "
                            + relative_path
                            + ":"
                            + str(line)
                            + ' | change "'
                            + ori_ss.strip()
                            + '" to "'
                            + ss.strip()
                            + ' " |\n'
                        )
                f.close()
                mlu_f.close()
        except:
            print(f"Failed in convertion of {file_path}!!!!")
            raise


print("# Cambricon PyTorch Model Migration Report")
print("Official PyTorch model scripts: ", input_path)
print("Cambricon PyTorch model scripts: ", mlu_path)
print("Migration Report: ", mlu_report)
