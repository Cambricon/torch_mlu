import argparse
import re
from collections import OrderedDict
import sys
import logging
import requests
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

# Note: you may need to update these pattern strings slightly when official html format change
API_PATTERN = re.compile(r"^.*[#\.](torch\..*?)\">")
SECTION_PATTERN = re.compile(r"<a href=\"(.*?)\.html#")
SECTION_MAP_PATTERN = re.compile(
    r"class=\"reference internal\" href=\"(.*)\.html\">([^<]*)<"
)
HREF_PATTERN = re.compile(r"<li><a href=\"(.*?)\"")
CURRENT_SECTION_PATTERN = re.compile(
    r"\"toctree-l1 current\"><a class=\"reference internal\" href=\"[\./]*([^\n]*).html\""
)

parser = argparse.ArgumentParser(
    description="Automatically output Pytorch Python API list in reST format to the screem "
    "from official docs of Pytorch."
)
parser.add_argument(
    "pytorch_version", help="the version of Pytorch you want to generate API list."
)

args = parser.parse_args()
docs_url = "https://pytorch.org/docs/{}/".format(args.pytorch_version)
index_html = requests.get(docs_url + "genindex.html").text


def section_search(line, matched_api, section2apis):
    href_match = HREF_PATTERN.search(line)
    generated_api_html = requests.get(docs_url + href_match.group(1)).text
    section_match = CURRENT_SECTION_PATTERN.search(generated_api_html)
    if section_match:
        section2apis[section_match.group(1)].add(matched_api)
    else:
        logging.warning(
            "Can not find which section this api "
            + matched_api
            + ' belong to, it will be save to "Others", please check the html file structure'
        )
        if "Others" not in section2apis:
            section2apis["Others"] = set()
        section2apis["Others"].add(matched_api)


# define some structures used to store the filtered results
section2apis = {}
section_map = OrderedDict()

logging.info("Start parsing table of contents ...")
index_html_lines = index_html.split("\n")
cur = 0
record_flg = False
for line in index_html_lines:
    line = line.lstrip()
    if "Python API" in line:
        record_flg = True
    if record_flg and line.startswith('<li class="toctree-l1"'):
        section_match = SECTION_MAP_PATTERN.search(line)
        if section_match:
            section_map[section_match.group(1)] = section_match.group(2)
            section2apis[section_match.group(1)] = set()
    if record_flg and line == "</ul>":
        break
    cur += 1

logging.info("Start parsing API ...")
for line in tqdm(index_html_lines[cur:]):
    line = line.lstrip()
    api_match = API_PATTERN.search(line)
    if api_match:
        matched_api = api_match.group(1)
        if matched_api.endswith("_"):
            matched_api = matched_api[:-1] + "\\_"
        if line.startswith('<li><a href="generated/'):
            section_search(line, matched_api, section2apis)
        elif line.startswith('<li><a href="'):
            section_match = SECTION_PATTERN.search(line)
            if not section_match:
                logging.error(
                    "Can not find section name, please check this line: " + line
                )
                sys.exit(1)
            if section_match.group(1) not in section2apis:
                section_search(line, matched_api, section2apis)
            else:
                section2apis[section_match.group(1)].add(matched_api)

logging.info("Start outputting API in reST List Table format...")
print("原生PyTorch社区API\n++++++++++++++++++++\n\nPyTorch API\n^^^^^^^^^^^^^^^^^^^^^^^^\n")
for k, v in section2apis.items():
    if k == "Others":
        print(k)
    else:
        print(section_map[k])
    print("====================================\n")
    if len(v) == 0:
        print()
        continue
    print(".. list-table::")
    if k == "Others":
        print("    :widths: 8 2 8")
    else:
        val = section_map[k]
        if val == "torch.cuda":
            print("    :widths: 4 4 2 4")
        elif val == "torch.amp":
            print("    :widths: 8 8 2 8")
        elif val == "torch.backends":
            print("    :widths: 5 5 1 3")
        elif val == "Understanding CUDA Memory Usage":
            print("    :widths: 4 4 2 4")
        elif (
            val == "torch.distributed.elastic"
            or val == "torch.distributions"
            or val == "DDP Communication Hooks"
            or val == "Quantization"
        ):
            print("    :widths: 8 2 4")
        else:
            print("    :widths: 8 2 8")
    print("    :header-rows: 1")
    print("    :align: center\n")
    print("    * - PyTorch API")
    if k != "Others" and (
        section_map[k] == "torch.cuda"
        or section_map[k] == "torch.amp"
        or section_map[k] == "torch.backends"
    ):
        print(
            "      - mlu对应API名称\n      - 是否支持\n      - 限制\n\n    * - "
            + "\n      -\n      -\n      -\n\n    * - ".join(sorted(v))
            + "\n      -\n      -\n      -\n\n"
        )
    elif k != "Others" and section_map[k] == "Understanding CUDA Memory Usage":
        print(
            "      - mlu对应API名称\n      - 是否支持\n      - 限制\n\n    * - "
            + "\n      -\n      -\n      -\n\n    * - ".join(sorted(v))
            + "\n      -\n      -\n      -\n\n"
        )
    else:
        print(
            "      - 是否支持\n      - 限制\n\n    * - "
            + "\n      -\n      -\n\n    * - ".join(sorted(v))
            + "\n      -\n      -\n\n"
        )
