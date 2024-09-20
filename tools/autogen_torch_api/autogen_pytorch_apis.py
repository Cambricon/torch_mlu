import argparse
import re
from collections import OrderedDict
import sys
import logging
import requests
from tqdm import tqdm
import json
import io
import yaml

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
    description="Automatically output Pytorch Python API list in yaml format to ./api_support_lists/ "
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
        section2apis[section_map[section_match.group(1)]][matched_api] = {
            "supported": False
        }
    else:
        logging.warning(
            "Can not find which section this api "
            + matched_api
            + ' belong to, it will be save to "Others", please check the html file structure'
        )
        if "Others" not in section2apis:
            section2apis["Others"] = {}
        section2apis["Others"][matched_api] = {"supported": False}


# define some structures used to store the filtered results
section2apis = OrderedDict()
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
            section2apis[section_map[section_match.group(1)]] = {}
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
            if section_match.group(1) not in section_map:
                section_search(line, matched_api, section2apis)
            else:
                section2apis[section_map[section_match.group(1)]][matched_api] = {
                    "supported": False
                }


def dict_to_list_format(d):
    result = []
    for key, value in d.items():
        if isinstance(value, dict):
            # Transfer dict to List
            entries = []
            for sub_key, sub_value in value.items():
                entry = {"api": sub_key}
                entry.update(sub_value)
                entries.append(entry)
            result.append({key: entries})
    return result


# Process yaml data
yaml_data = dict_to_list_format(section2apis)
yaml_string = io.StringIO()
yaml.dump(
    yaml_data,
    yaml_string,
    sort_keys=False,
    default_flow_style=False,
    allow_unicode=True,
)
output_yaml = yaml_string.getvalue()
output_yaml = output_yaml.replace("\\", "")
yaml_string = io.StringIO(output_yaml)


yaml_content = yaml_string.getvalue()

# Add empty lines
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

with open("new_api_support_list.yaml", "w", encoding="utf-8") as yaml_file:
    yaml_file.write(yaml_content_with_blank_lines)
