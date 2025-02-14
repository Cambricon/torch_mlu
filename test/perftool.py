import argparse
from collections import OrderedDict
import os
import json
import pandas as pd
import numpy as np

cur_dir = os.path.dirname(os.path.abspath(__file__))

base_report_dir = cur_dir + "/torch_ops/ops_kernel_report_baseline"
report_dir = cur_dir + "/torch_ops/ops_kernel_report"
result_dir = cur_dir + "/torch_ops/ops_kernel_cmp_result"

base_e2e_json_path = cur_dir + "/ops_E2E_baseline.json"
e2e_json_path = cur_dir + "/ops_E2E.json"
e2e_result_path = cur_dir + "/ops_E2E_cmp_result.csv"

base_hardware_json_path = cur_dir + "/ops_hardware_time_baseline.json"
hardware_json_path = cur_dir + "/ops_hardware_time.json"
hardware_result_path = cur_dir + "/ops_hardware_time_cmp_result.csv"

# names of col1 and col2 of result csv
KEY1, KEY2 = "Baseline", "Exp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Perftool to compare Op's kernel calls"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Print the perf comparison results file-by-file.",
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs=1,
        default=[],
        help="Specify one testfile's basename(e.g., test_abs) to execute perf comparison.",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="All files will be selected to execute perf comparison.",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Whether to execute op's E2E-Time comparison.",
    )
    parser.add_argument(
        "--hardware",
        action="store_true",
        help="Whether to execute op's Hardware-Time comparison.",
    )
    parser.add_argument(
        "--base_report_dir",
        default=base_report_dir,
        help="Kernel launch number baseline directory.",
    )
    parser.add_argument(
        "--report_dir",
        default=report_dir,
        help="Current Kernel launch number directory.",
    )
    parser.add_argument(
        "--result_dir",
        default=result_dir,
        help="Kernel launch number comparison result directory.",
    )
    return parser.parse_args()


def processing_case(df, df_base):
    if df.empty:
        if df_base.empty:
            return None
        df["Name"] = None
        df["# of Calls"] = None
    if df_base.empty:
        df_base["Name"] = None
        df_base["# of Calls"] = None
    df_base = df_base.set_index("Name")  # type: ignore
    df = df.set_index("Name")  # type: ignore
    df_res = pd.concat([df_base, df], axis=1, keys=[KEY1, KEY2])  # type: ignore
    df_res.fillna(0, inplace=True)
    df_res["Diff of Calls"] = df_res[KEY2, "# of Calls"] - df_res[KEY1, "# of Calls"]  # type: ignore
    df_res = df_res[df_res["Diff of Calls"] != 0]
    if df_res.empty:  # type: ignore
        return None
    df_res.sort_values(by="Diff of Calls", ascending=False, inplace=True)  # type: ignore
    return df_res


def processing_file(file_name: str, options):
    """
    Returns:
        ``None``, if device_count not equal between exp and baseline
        ``empty DataFrame``, if no difference between exp and baseline
        ``DataFrame``, including the difference
    """
    if options.verbose:
        print("Selected file: " + file_name + " to execute perf comparison...")
    file_basename = file_name.rsplit(".", maxsplit=1)[0]
    with open(base_report_dir + "/" + file_name, "r") as infile1:
        base_jsonstr = json.loads(infile1.read())
        base_num_cards, base_cases = (
            base_jsonstr["device_count"],
            base_jsonstr[file_basename],
        )
    with open(report_dir + "/" + file_name, "r") as infile2:
        jsonstr = json.loads(infile2.read())
        num_cards, cases = jsonstr["device_count"], jsonstr[file_basename]
    if num_cards != base_num_cards:
        print(
            "Device count not equal between exp and baseline: "
            + " baseline is "
            + str(base_num_cards)
            + ", but exp is "
            + str(num_cards)
            + "! Skip comparison for file: "
            + file_name
        )
        return None
    df_base_cases = [
        pd.json_normalize(case, record_path=["Events"]) for case in base_cases
    ]
    df_cases = [pd.json_normalize(case, record_path=["Events"]) for case in cases]
    base_casenames_lst = [case["CaseName"] for case in base_cases]
    casenames_lst = [case["CaseName"] for case in cases]
    interset = set(casenames_lst).intersection(set(base_casenames_lst))
    df_res_lst = []
    keys_res = []
    for i, df_case in enumerate(df_cases):
        if casenames_lst[i] in interset:
            idx = base_casenames_lst.index(casenames_lst[i])
            df_case_res = processing_case(df_case, df_base_cases[idx])
            if df_case_res is not None:
                df_res_lst.append(df_case_res)
                keys_res.append(casenames_lst[i])
        else:
            df_res_lst.append(pd.DataFrame({"Name": ["New Case!"]}).set_index("Name"))
            keys_res.append(casenames_lst[i])
    if not df_res_lst:
        print("No difference with the baseline for file: " + file_name)
        return pd.DataFrame()
    df_res = pd.concat(df_res_lst, keys=keys_res, names=["Case", "Name"])
    df_res.fillna(0, inplace=True)
    df_res = df_res.astype("int32")  # type: ignore
    if options.verbose:
        print(df_res)
    # Save to csv
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df_res.to_csv(result_dir + "/" + file_basename + ".csv", encoding="utf-8")
    print(
        "Save comparison result to "
        + result_dir
        + "/"
        + file_basename
        + ".csv successfully."
    )
    return df_res


def processing_all_files(options):
    base_files = [
        name for name in os.listdir(base_report_dir) if name.endswith(".json")
    ]
    files = [name for name in os.listdir(report_dir) if name.endswith(".json")]
    interset = set(files).intersection(set(base_files))
    df_res_lst = []
    keys_res = []
    for file in files:
        if file in interset:
            df_file = processing_file(file, options)
            if df_file is None:
                df_res_lst.append(
                    pd.DataFrame(
                        {
                            "Case": ["Device count not equal, skip comparison!"],
                            "Name": ["Skip Case!"],
                        }
                    ).set_index(["Case", "Name"])
                )
                keys_res.append(file.rsplit(".", maxsplit=1)[0] + ".py")
            elif not df_file.empty:
                df_res_lst.append(df_file)
                keys_res.append(file.rsplit(".", maxsplit=1)[0] + ".py")
        else:
            df_res_lst.append(
                pd.DataFrame({"Case": ["New File!"], "Name": ["New Case!"]}).set_index(
                    ["Case", "Name"]
                )
            )
            keys_res.append(file.rsplit(".", maxsplit=1)[0] + ".py")
    if not df_res_lst:
        print("No difference with the baseline for all files!")
        return
    df_res = pd.concat(df_res_lst, keys=keys_res, names=["File", "Case", "Name"])
    df_res.fillna(0, inplace=True)
    df_res = df_res.astype("int32")  # type: ignore
    # Save to csv
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    df_res.to_csv(result_dir + "/all_in_one.csv", encoding="utf-8")
    print(
        "Save all comparison results to " + result_dir + "/all_in_one.csv successfully."
    )


def ops_time_compare_file(dic, dic_base, type_str):
    def dic2df(dic):
        dic_new = OrderedDict()
        dic_new["Case"] = list(dic.keys())
        dic_new[type_str] = [dic[name] for name in dic_new["Case"]]
        return pd.DataFrame(dic_new)

    df = (
        dic2df(dic).set_index("Case")
        if dic
        else pd.DataFrame({"Case": list(dic_base.keys()), type_str: None}).set_index(
            "Case"
        )
    )
    df_base = dic2df(dic_base).set_index("Case")
    df_res = pd.concat([df_base, df], axis=1, keys=[KEY1, KEY2])  # type: ignore
    df_res["Diff of " + type_str] = df_res[KEY2, type_str] - df_res[KEY1, type_str]  # type: ignore
    df_res["Relative diff of Time"] = df_res["Diff of " + type_str] / (df_res[KEY1, type_str].replace(0, np.nan))  # type: ignore
    return df_res


def ops_time_compare_all_files(
    time_json_path, base_time_json_path, result_path, type_str="E2E-Time(s)"
):
    with open(time_json_path, "r") as f1:
        dic_e2e = json.loads(f1.read())
    with open(base_time_json_path, "r") as f2:
        dic_base_e2e = json.loads(f2.read())
    filename_lst, base_filename_lst = list(dic_e2e.keys()), list(dic_base_e2e.keys())
    interset = set(filename_lst).intersection(set(base_filename_lst))
    df_res_lst = []
    keys_res = []
    for filename in filename_lst:
        keys_res.append(filename + ".py")
        if filename in interset:
            df_file = ops_time_compare_file(
                dic_e2e[filename], dic_base_e2e[filename], type_str
            )
            df_res_lst.append(df_file)
        else:
            df_res_lst.append(pd.DataFrame({"Case": ["New File!"]}).set_index("Case"))
    for filename in base_filename_lst:
        if filename not in interset:
            df_file = ops_time_compare_file(None, dic_base_e2e[filename], type_str)
            df_res_lst.append(df_file)
            keys_res.append(filename + ".py")
    df_res = pd.concat(df_res_lst, keys=keys_res, names=["File", "Case"])
    df_res.sort_values(by="File", ascending=True, inplace=True)
    # Save to csv
    df_res.to_csv(result_path, encoding="utf-8", float_format="%.5f")
    print("Save " + type_str + " comparison results successfully.")


def main():
    options = parse_args()
    if options.e2e:
        print("Run op's E2E-Time comparison...")
        ops_time_compare_all_files(
            e2e_json_path, base_e2e_json_path, e2e_result_path, "E2E-Time(s)"
        )

    if options.hardware:
        print("Run op's Hardware-Time comparison...")
        ops_time_compare_all_files(
            hardware_json_path,
            base_hardware_json_path,
            hardware_result_path,
            "Hardware-Time(ms)",
        )

    if options.all:
        print("Run all files to execute perf comparison...")
        processing_all_files(options)

    if options.file:
        print("Run single file to execute perf comparison...")
        processing_file(options.file[0] + ".json", options)


if __name__ == "__main__":
    pd.set_option("display.max_rows", 1000)
    main()
