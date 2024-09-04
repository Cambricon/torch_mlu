import json
import os

ABS_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
json_file = os.path.join(ABS_DIR_PATH, "build.property")
output = "dependency.txt"
output2 = "dependency2.txt"
generate = "release.txt"

with open(json_file, encoding="utf-8") as json_data, open(
    output, "w", encoding="utf-8"
) as output_file, open(output2, "w", encoding="utf-8") as output_file2, open(
    generate, "w", encoding="utf-8"
) as release_file:
    data = json.load(json_data)
    for key in data["build_requires"]:
        value = data["build_requires"][key]
        if key == "cndali":
            continue
        elif key == "benchmark":
            continue
        output_file.write(f"{key}:{value[0]}:{value[1]}\n")
        if key == "cnnlextra":
            key = "cnnl_extra"
        output_file2.write(f"{key}:{value[0]}:{value[1]}\n")
    release_file.write(f"pytorch:release:{data['version']}\n")
