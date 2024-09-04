import os
import sys
import ruamel.yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + "/../")
from upstream_code_check import *

"""
This is a test file for updating line number and commit id.
The dummy data is stored in test_associated.yaml,
and the file will be updated with the latest line number and commit id.

real_value_for_test.yaml stores the true value of line number.

Details of native changes for testing:

https://github.com/pytorch/pytorch/pull/122015
https://github.com/pytorch/pytorch/pull/121089
https://github.com/pytorch/pytorch/pull/121070
https://github.com/pytorch/pytorch/pull/119713
https://github.com/pytorch/pytorch/pull/121106
https://github.com/pytorch/pytorch/pull/120969

"""


def test_with_dummy_data():
    commit_new = "24944f6717c087416b7f4a8bef4adee087624336"
    repo_path = os.environ.get("PYTORCH_HOME")
    assert repo_path, "PYTORCH_HOME should be set."
    metadata_file = os.path.dirname(os.path.abspath(__file__)) + "/test_associated.yaml"
    result_file = (
        os.path.dirname(os.path.abspath(__file__)) + "/real_value_for_test.yaml"
    )
    folder_path = os.path.dirname(os.path.abspath(__file__)) + "/test_diff_result"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    metadata = load_metadata(metadata_file)
    check_associated_code_changes_all(
        repo_path, commit_new, metadata, folder_path, True, 0
    )
    update_metadata(metadata_file, metadata)
    compare_yaml_files(metadata_file, result_file)


def compare_yaml_files(file_1, file_2):
    data_1 = load_metadata(file_1)
    data_2 = load_metadata(file_2)

    for entry_1, entry_2 in zip(data_1, data_2):
        file_path = entry_1["file"]
        commit_1 = entry_1["commit_id"]
        commit_2 = entry_2["commit_id"]
        assert (
            commit_1 == commit_2
        ), "commit id not equal with file: {}, Please check.".format(file_path)

        source_info_1 = entry_1["source_info"]
        source_info_2 = entry_2["source_info"]
        for info_1, info_2 in zip(source_info_1, source_info_2):
            if "associated_code" not in info_1:
                continue
            associated_code_1 = info_1["associated_code"]
            associated_code_2 = info_2["associated_code"]
            for line_code_1, line_code_2 in zip(associated_code_1, associated_code_2):
                start_line_1 = line_code_1["start_line"]
                start_line_2 = line_code_2["start_line"]
                assert (
                    start_line_1 == start_line_2
                ), "line number inference error with file: {}, Please check.".format(
                    file_path
                )
                end_line_1 = line_code_1["end_line"]
                end_line_2 = line_code_2["end_line"]
                assert (
                    start_line_1 == start_line_2
                ), "line number inference error with file: {}, Please check.".format(
                    file_path
                )


if __name__ == "__main__":
    test_with_dummy_data()
    print("Test passed")
