import os
import re
import ruamel.yaml
import argparse
import subprocess

yaml = ruamel.yaml.YAML()


def load_metadata(metadata_file):
    with open(metadata_file, "r") as file:
        metadata = yaml.load(file)
    return metadata


def update_metadata(metadata_file, metadata):
    with open(metadata_file, "w") as file:
        metadata = yaml.dump(metadata, file)


def get_diff_blocks(diff_output):
    """
    Split the information obtained by git diff into individual diff blocks.
    Return a list named diff_blocks, diff_blocks[0] is diff head information,
    and diff_blocks[1:] are  diff blocks.
    """
    diff_blocks = []
    current_block = []
    for line in diff_output.split("\n"):
        if line.startswith("@@"):
            if current_block:
                diff_blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        diff_blocks.append("\n".join(current_block))

    return diff_blocks


def get_changed_lines(diff_block):
    """
    Get a list of changed line numbers from a diff block.
    Return a list named lines, which contains all changed line numbers.
    Such as :
    @@ -194,4 +194,4 , add 194,195,196 and 197 to the list.
    @@ -100, +110,8 , add 100 to the list.
    @@ -200,0 +220,20 , add 200 to the list.
    """

    lines = []
    diff_info = diff_block.split()[1]
    diff_info_line = diff_info[1:].split(",")
    assert 1 <= len(diff_info_line) <= 2, "assert error"
    start_line = int(diff_info_line[0])
    length = 1
    if len(diff_info_line) == 2 and int(diff_info_line[1]) != 0:
        length = int(diff_info_line[1])
    lines = list(range(start_line, start_line + length))
    return lines


def auto_update_line_number_one_associated(diff_output, start_line, end_line):
    """
    Use the information of the diff block to update the line number.
    We update with three pieces of information like "@@ ... ... @@".
    We need to get the "@@ information" of the left adjacent start_line: @@ -25,3 +30,10 @@,
    and the "@@ information" adjacent to the end_line: @@ -75,10 +85,10 @@, @@ -100,0 +110,10 @@.
    Through these three "@@ information", we can infer the last line number of the associated code.

    # line 0  -----------------------------------------------------------
    ...
    @@ -25,3 +30,10 @@
    ...
    # start_line ---------
    ...
    @@ -50,10 +60,10 @@
    ...
    @@ -75,10 +85,10 @@
    ...
    ...
    # end_line -----------
    ...
    @@ -100,0 +110,10 @@
    ...
    # line -1  -----------------------------------------------------------

    ```
    new_start_line = start_line + (30 - 25 + 10 - 3)
    new_end_line = end_line + (110 - 100 + 10 - 0)  or  end_line + (85 - 75 + 10 - 10)
    ```

    Similar calculations for different situations

    note:
    auto_update_line_number is only recommended to be used when file changes do not affect the
    associated code, that is, when there is no diff block between start_line and end_line.
    In other cases, it is recommended to manually update start_line and end_line.

    """

    diff_blocks = get_diff_blocks(diff_output)

    x_line_old, y_line_old, z_line_old = None, None, None
    x_line_new, y_line_new, z_line_new = None, None, None
    x_decrease, x_increase = None, None
    z_decrease, z_increase = None, None
    y_decrease, y_increase = None, None
    for diff_block in diff_blocks[1:]:
        line_old_info = list(map(int, diff_block.split()[1][1:].split(",")))
        line_new_info = list(map(int, diff_block.split()[2][1:].split(",")))
        line_old = line_old_info[0]
        if line_old <= start_line:
            x_line_old = line_old
            x_line_new = line_new_info[0]
            x_decrease = 1 if len(line_old_info) == 1 else line_old_info[1]
            x_increase = 1 if len(line_new_info) == 1 else line_new_info[1]
        elif start_line < line_old < end_line:
            z_line_old = line_old
            z_line_new = line_new_info[0]
            z_decrease = 1 if len(line_old_info) == 1 else line_old_info[1]
            z_increase = 1 if len(line_new_info) == 1 else line_new_info[1]
        elif line_old >= end_line:
            y_line_old = line_old
            y_line_new = line_new_info[0]
            y_decrease = 1 if len(line_old_info) == 1 else line_old_info[1]
            y_increase = 1 if len(line_new_info) == 1 else line_new_info[1]
            break

    def if_zero(line_old, line_new, decrease, increase):
        line_old = line_old + 1 if not decrease else line_old
        line_new = line_new + 1 if not increase else line_new
        return line_old, line_new

    def at_bound(
        line_old,
        line_decrease,
        line_new,
        line_increase,
        line_org,
        if_y=False,
        if_end=False,
    ):
        if line_old <= line_org <= line_old + line_decrease - 1:
            if not line_decrease:
                return line_new - 1
            if not line_increase:
                return line_new
            if if_end:
                return line_new + line_increase - 1
            else:
                return line_new
        elif if_y:
            line_old, line_new = if_zero(
                line_old, line_new, line_decrease, line_increase
            )
            return line_org + (line_new - line_old)
        else:
            line_old, line_new = if_zero(
                line_old, line_new, line_decrease, line_increase
            )
            return line_org + (line_new - line_old + line_increase - line_decrease)

    if x_line_old and y_line_old:
        new_start_line = at_bound(
            x_line_old, x_decrease, x_line_new, x_increase, start_line
        )
        new_end_line = at_bound(
            y_line_old,
            y_decrease,
            y_line_new,
            y_increase,
            end_line,
            if_y=True,
            if_end=True,
        )
        return new_start_line, new_end_line
    elif x_line_old:
        if z_line_old:
            new_start_line = at_bound(
                x_line_old, x_decrease, x_line_new, x_increase, start_line
            )
            new_end_line = at_bound(
                z_line_old, z_decrease, z_line_new, z_increase, end_line, if_end=True
            )
            return new_start_line, new_end_line
        else:
            new_start_line = at_bound(
                x_line_old, x_decrease, x_line_new, x_increase, start_line
            )
            new_end_line = at_bound(
                x_line_old, x_decrease, x_line_new, x_increase, end_line, if_end=True
            )
            return new_start_line, new_end_line
    elif y_line_old:
        new_start_line = start_line
        new_end_line = at_bound(
            y_line_old,
            y_decrease,
            y_line_new,
            y_increase,
            end_line,
            if_y=True,
            if_end=True,
        )
        return new_start_line, new_end_line
    elif z_line_old:
        new_start_line = start_line
        new_end_line = at_bound(
            z_line_old, z_decrease, z_line_new, z_increase, end_line, if_end=True
        )
        return new_start_line, new_end_line
    else:
        return start_line, end_line


def detect_code_changes(diff_output, start_line, end_line):
    diff_blocks = get_diff_blocks(diff_output)
    corr_diff_blocks = ""
    has_corr = False
    for diff_block in diff_blocks[1:]:
        lines = get_changed_lines(diff_block)
        has_corr_ = False
        for line_number in lines:
            if start_line <= line_number <= end_line:
                has_corr_ = True
                has_corr = True
                break
        if has_corr_:
            corr_diff_blocks += f"{diff_block}\n"

    return diff_blocks[0], corr_diff_blocks, has_corr


# def check_associated_code_changes(file_path, source_file, repo_path, commit_old,
#                                  commit_new, associated_code, if_overide,
#                                  check_level):
def check_associated_code_changes(
    file_path, repo_path, commit_old, commit_new, if_overide, check_level, info
):
    corr_diff_blocks_all = ""
    diff_head = ""
    associated_code = ""
    source_file = info["source_file"]
    try:
        output = subprocess.check_output(
            f"git -C {repo_path} diff --unified=0 {commit_old}..{commit_new} {source_file}",
            shell=True,
        )
        if not output:
            return corr_diff_blocks_all

    except subprocess.CalledProcessError:
        print(
            "Error: Failed to get code diff with: \n{}".format(
                f"git -C {repo_path} diff --unified=0 {commit_old}..{commit_new} {source_file}"
            )
        )
        exit(1)

    diff_output = output.decode("utf-8")

    if check_level:
        return diff_output

    line_max_old = int(
        subprocess.check_output(
            f"git -C {repo_path} show {commit_old}:{source_file} | wc -l", shell=True
        )
    )
    line_max_new = int(
        subprocess.check_output(
            f"git -C {repo_path} show {commit_new}:{source_file} | wc -l", shell=True
        )
    )

    if "associated_code" in info:
        associated_code = info["associated_code"]
    else:
        return diff_output

    for line_code in associated_code:
        start_line = line_code["start_line"]
        end_line = line_code["end_line"]
        if end_line == -1 and start_line == 0:
            return diff_output
        elif end_line == -1:
            end_line = line_max_old

        diff_head, corr_diff_blocks, has_corr = detect_code_changes(
            diff_output, start_line, end_line
        )
        if if_overide:
            new_start_line, new_end_line = auto_update_line_number_one_associated(
                diff_output, start_line, end_line
            )
            line_code["start_line"] = new_start_line
            line_code["end_line"] = new_end_line if new_end_line < line_max_new else -1
        if has_corr:
            corr_diff_blocks_all += corr_diff_blocks
    if corr_diff_blocks_all and diff_head:
        corr_diff_blocks_all = diff_head + "\n" + corr_diff_blocks_all
    return corr_diff_blocks_all


def check_associated_code_changes_all(
    repo_path, commit_new, metadata, folder_path, if_overide, check_level
):
    for entry in metadata:
        file_path = entry["file"]
        commit_old = entry["commit_id"]
        source_info = entry["source_info"]
        diff_need_to_write = ""
        for info in source_info:
            corr_diff_blocks_all = check_associated_code_changes(
                file_path,
                repo_path,
                commit_old,
                commit_new,
                if_overide,
                check_level,
                info,
            )
            if corr_diff_blocks_all:
                diff_need_to_write += f"{corr_diff_blocks_all}\n"
        entry["commit_id"] = commit_new
        if diff_need_to_write:
            file_name = re.sub(r"[./]", "_", file_path) + ".diff"
            file_path_diff = os.path.join(folder_path, file_name)
            print(
                "The code associated with {} has changed, please check it. Details: {}".format(
                    file_path, file_path_diff
                )
            )
            with open(file_path_diff, "w") as f:
                f.write(diff_need_to_write)


def check_associated_code_changes_single(
    repo_path, commit_new, metadata, check_file, folder_path, if_overide, check_level
):
    has_file = False
    for entry in metadata:
        file_path = entry["file"]
        if file_path == check_file:
            has_file = True
            commit_old = entry["commit_id"]
            source_info = entry["source_info"]
            diff_need_to_write = ""
            for info in source_info:
                corr_diff_blocks_all = check_associated_code_changes(
                    file_path,
                    repo_path,
                    commit_old,
                    commit_new,
                    if_overide,
                    check_level,
                    info,
                )
                if corr_diff_blocks_all:
                    diff_need_to_write += f"{corr_diff_blocks_all}\n"
            entry["commit_id"] = commit_new
            if diff_need_to_write:
                file_name = re.sub(r"[./]", "_", file_path) + ".diff"
                file_path_diff = os.path.join(folder_path, file_name)
                print(
                    "The code associated with {} has changed, please check it. Details: {}".format(
                        file_path, file_path_diff
                    )
                )
                with open(file_path_diff, "w") as f:
                    f.write(diff_need_to_write)
            break

    assert has_file, "check_file not exit in yaml file, please check or add"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for check tool")
    parser.add_argument("-c", "--commit", type=str, default=None, help="Target commit")
    parser.add_argument(
        "-p", "--repo_path", type=str, default=None, help="PyTorch repo path"
    )
    parser.add_argument(
        "-m",
        "--metadata_file",
        type=str,
        default="code_associated.yaml",
        help="associated code info between catch and pytorch",
    )
    parser.add_argument(
        "-f", "--check_file", type=str, default=None, help="Which file check"
    )
    parser.add_argument(
        "-w", "--write", action="store_true", help="Update metadata_file"
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=0,
        help="The granularity of check source file changes, 0: code level, 1: file level.",
    )

    args = parser.parse_args()
    commit_new = args.commit
    repo_path = args.repo_path
    if not repo_path:
        repo_path = os.environ.get("PYTORCH_HOME")
    assert repo_path, "PYTORCH_HOME should be set, or give a path with '--repo_path'."
    metadata_file = args.metadata_file
    check_file = args.check_file
    if_overide = args.write
    check_level = args.level
    assert not (
        if_overide and check_level
    ), "File level check does not support Update metadata file."
    metadata = load_metadata(metadata_file)

    folder_path = os.path.dirname(os.path.abspath(__file__)) + "/diff_result"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    if check_file:
        check_associated_code_changes_single(
            repo_path,
            commit_new,
            metadata,
            check_file,
            folder_path,
            if_overide,
            check_level,
        )
    else:
        check_associated_code_changes_all(
            repo_path, commit_new, metadata, folder_path, if_overide, check_level
        )

    if if_overide:
        update_metadata(metadata_file, metadata)

    if not os.listdir(folder_path):
        os.rmdir(folder_path)
