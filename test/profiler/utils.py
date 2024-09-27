import csv
import json


def get_kernel_total_info_from_json(json_file):
    device_time = 0.0
    kernel_count = 0
    with open(json_file, "r") as f:
        data = json.load(f)
        for event in data["traceEvents"]:
            if event.get("cat") == "kernel":
                kernel_count += 1
                device_time += event["dur"]

    return device_time, kernel_count


def get_kernel_total_info_from_kernel_details_csv(csv_file):
    device_time = 0.0
    kernel_count = 0
    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        # ['Thread Id', 'Correlation Id', 'Kernel Name', 'Operator',
        # 'Operator Input Shapes', 'Operator Input Type', 'Start Time', 'Duration(us)', ... ]
        header = next(reader)
        for row in reader:
            device_time += float(row[7])
            kernel_count += 1

    return device_time, kernel_count, header


def get_kernel_total_info_from_kernel_statistic_csv(csv_file):
    device_time = 0.0
    kernel_count = 0
    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        # ['Kernel Name', 'Count', 'Total Time(us)', ... ]
        header = next(reader)
        for row in reader:
            device_time += float(row[2])
            kernel_count += int(row[1])

    return device_time, kernel_count, header


def get_kernel_total_info_from_op_kernel_statistic_csv(csv_file):
    device_time = 0.0
    kernel_count = 0
    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        # ['Kernel Name', 'Operator', 'Count', 'Total Time(us)', ... ]
        header = next(reader)
        for row in reader:
            device_time += float(row[3])
            kernel_count += int(row[2])

    return device_time, kernel_count, header
