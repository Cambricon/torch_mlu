from __future__ import print_function

import sys
import json
import os
import io
import csv
import copy
import time
import tempfile
import unittest
from unittest.mock import patch
import logging
import pickle

import torch
import torch_mlu
import torch.nn as nn

from utils import (
    get_kernel_total_info_from_json,
    get_kernel_total_info_from_kernel_details_csv,
    get_kernel_total_info_from_kernel_statistic_csv,
    get_kernel_total_info_from_op_kernel_statistic_csv,
)

from torch.testing._internal.common_utils import (
    TemporaryFileName,
    TemporaryDirectoryName,
)

from torch.profiler import (
    profile,
    record_function,
    supported_activities,
    DeviceType,
    ProfilerAction,
    ProfilerActivity,
)

from torch.autograd.profiler import profile as _profile
from torch.autograd.profiler import record_function as _record_function
from torch.autograd import _disable_profiler

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

TEST_MULTIMLU = torch.is_mlu_available() and torch.mlu.device_count() >= 2


class TestProfiler(TestCase):
    # @unittest.skip("not test")
    def payload(self, use_mlu=False):
        x = torch.randn(10, 10)
        if use_mlu:
            x = x.to("mlu")
        y = torch.randn(10, 10)
        if use_mlu:
            y = y.to("mlu")
        z = torch.mm(x, y)
        z = z + y
        if use_mlu:
            z = z.cpu()

    def check_csv_file(self, file_path):
        try:
            with open(file_path, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                if not header:
                    return False
                for row in reader:
                    if len(row) != len(header):
                        return False
            return True
        except Exception:
            return False

    # @unittest.skip("not test")
    def test_kineto_profiler_api(self):
        called_num = [0]
        use_mlu = torch.profiler.ProfilerActivity.MLU in supported_activities()
        with profile(activities=supported_activities()):
            self.payload(use_mlu=use_mlu)

        def trace_handler(p):
            output = p.key_averages().table(
                sort_by="self_mlu_time_total" if use_mlu else "self_cpu_time_total",
                row_limit=-1,
            )
            called_num[0] += 1

        with profile(
            activities=supported_activities(),
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2),
            on_trace_ready=trace_handler,
        ) as p:
            for idx in range(8):
                self.payload(use_mlu=use_mlu)
                p.step()
        self.assertEqual(called_num[0], 2)

        # case without schedule
        with profile(activities=supported_activities()) as p:
            self.payload(use_mlu=use_mlu)
            self.payload(use_mlu=use_mlu)
        output = p.key_averages().table(
            sort_by="self_mlu_time_total" if use_mlu else "self_cpu_time_total",
            row_limit=-1,
        )

        test_schedule = torch.profiler.schedule(
            skip_first=2, wait=1, warmup=1, active=2, repeat=2
        )
        test_schedule_expected_outputs = [
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.WARMUP,
            ProfilerAction.RECORD,
            ProfilerAction.RECORD_AND_SAVE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
            ProfilerAction.NONE,
        ]
        for step in range(len(test_schedule_expected_outputs)):
            self.assertEqual(test_schedule(step), test_schedule_expected_outputs[step])

    # @unittest.skip("not test")
    def test_flops_memory_recordshapes(self):
        model = torch.nn.Sequential(
            nn.Conv2d(16, 33, 18),
            nn.ReLU(),
            nn.Linear(243, 243),
            nn.ReLU(),
        )
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ],
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
        ) as kineto_profiler:
            inputs = torch.randn(40, 16, 18, 260).to("mlu")
            model = model.to("mlu")
            model(inputs)
        profiler_output = kineto_profiler.key_averages().table(
            sort_by="self_mlu_time_total", row_limit=-1
        )
        record_shape_success = False
        for evt in kineto_profiler.events():
            if "aten::conv2d" in evt.name:
                if (
                    [40, 16, 18, 260] in evt.input_shapes
                    and [33, 16, 18, 18] in evt.input_shapes
                    and [33] in evt.input_shapes
                ):
                    record_shape_success = True
        self.assertIn("FLOPs", profiler_output)
        self.assertIn("MLU Mem", profiler_output)
        self.assertIn("CPU Mem", profiler_output)
        self.assertTrue(record_shape_success)

    # @unittest.skip("not test")
    def test_tensorboard_trace_handler(self):
        use_mlu = torch.profiler.ProfilerActivity.MLU in supported_activities()
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.MLU] if use_mlu else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    self.payload(use_mlu=use_mlu)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-3:], ["pt", "trace", "json"])
                file_num += 1
            self.assertEqual(file_num, 3)

        # test case for gzip file format
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
                + ([torch.profiler.ProfilerActivity.MLU] if use_mlu else []),
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    dname, use_gzip=True
                ),
            ) as p:
                for _ in range(18):
                    self.payload(use_mlu=use_mlu)
                    p.step()

            self.assertTrue(os.path.exists(dname))
            file_num = 0
            for file_name in os.listdir(dname):
                parts = file_name.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-5].isdigit() and int(parts[-5]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-4:], ["pt", "trace", "json", "gz"])
                file_num += 1
            self.assertEqual(file_num, 3)

    # @unittest.skip("not test")
    def test_profiler_metadata(self):
        t1, t2 = torch.ones(1), torch.ones(1)
        with profile() as prof:
            torch.add(t1, t2)
            prof.add_metadata("test_key1", "test_value1")
            prof.add_metadata_json("test_key2", "[1,2,3]")

        with TemporaryFileName(mode="w+") as fname:
            prof.export_chrome_trace(fname)
            with io.open(fname, "r") as f:
                trace = json.load(f)
                assert "test_key1" in trace
                assert trace["test_key1"] == "test_value1"
                assert "test_key2" in trace
                assert trace["test_key2"] == [1, 2, 3]

    # @unittest.skip("not test")
    def test_record_function(self):
        with profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
            with record_function("label"):
                x = torch.randn(10, 10).to("mlu")
                y = x * 2 + 4  # pylint: disable=W0612

        last_end = 0
        labels = ["label"]
        has_label = False
        for evt in p.events():
            if evt.name == "label":
                has_label = True
        self.assertTrue(has_label)

    @unittest.skipIf(not TEST_MULTIMLU, "Multiple MLUs needed")
    def test_profiler_multi_mlu(self):
        x = torch.randn(10, 10).to("mlu:0")
        x2 = torch.randn(10, 10).to("mlu:1")

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,  # 开启MLU profiler
            ],
        ) as p:
            y = x * 2 + 4  # pylint: disable=W0612
            y2 = x2 * 2 + 4  # pylint: disable=W0612
        found_mul = False
        found_add = False
        found_mlu0 = False
        found_mlu1 = False
        for evt in p.events():
            if "mul" in evt.name:
                found_mul = True
            if "add" in evt.name:
                found_add = True
            if evt.device_type == DeviceType.PrivateUse1:
                if evt.device_index == 0:
                    found_mlu0 = True
                if evt.device_index == 1:
                    found_mlu1 = True
        self.assertTrue(found_mul)
        self.assertTrue(found_add)
        self.assertTrue(found_mlu0)
        self.assertTrue(found_mlu1)

    # @unittest.skip("not test")
    def test_high_level_trace(self):
        """Checks that python side high level events are recorded."""

        class RepeatedDataset(torch.utils.data.Dataset):
            def __init__(self, N, D_in, D_out):
                self.N = N
                self.x = torch.randn(N, D_in)
                self.y = torch.randn(N, D_out)

            def __len__(self):
                return self.N

            def __getitem__(self, idx):
                return self.x, self.y

        class TwoLayerNet(torch.nn.Module):
            def __init__(self, D_in, H, D_out):
                super(TwoLayerNet, self).__init__()
                self.linear1 = torch.nn.Linear(D_in, H)
                self.linear2 = torch.nn.Linear(H, D_out)

            def forward(self, x):
                h_relu = self.linear1(x).clamp(min=0)
                y_pred = self.linear2(h_relu)
                return y_pred

        class CustomSGD(torch.optim.SGD):
            def __init__(self, *args, **kwargs):
                super(CustomSGD, self).__init__(*args, **kwargs)

        def train(prof=None):
            for _, data in enumerate(dataloader):
                x, y = data[0], data[1]
                y_pred = model(x.to("mlu"))
                loss = criterion(y_pred, y.to("mlu"))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if prof != None:
                    prof.step()

        N, D_in, H, D_out = 8, 10, 5, 2
        model = TwoLayerNet(D_in, H, D_out).to("mlu")
        criterion = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        ds = RepeatedDataset(N, D_in, D_out)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=1)

        try:
            train()
        except Exception:
            self.assertTrue(False, "Expected no exception without profiling.")

        # Create multiple instances, expect each func is hooked only one time.
        # Nested wrappers(repeated patching) will make following test fail.
        optimizer_duplicate = torch.optim.SGD(model.parameters(), lr=1e-4)
        dataloader_duplicate = torch.utils.data.DataLoader(ds, batch_size=1)

        def judge(expected_event_count, prof):
            actual_event_count = {}
            for e in prof.events():
                if "#" in e.name:
                    key = e.name
                    if key in expected_event_count.keys():
                        actual_event_count[key] = (
                            actual_event_count.setdefault(key, 0) + 1
                        )
            for key, count in expected_event_count.items():
                self.assertTrue(
                    (key in actual_event_count.keys())
                    and (count == actual_event_count[key])
                )

        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,  # 开启MLU profiler
            ],
            schedule=torch.profiler.schedule(
                wait=0,  # 等待wait个steps再启动profiler
                warmup=0,  # 前warmup个steps数据只采集，不记录
                active=10,
            ),  # 实际最大采集active个steps数据
            profile_memory=True,
            record_shapes=True,  # 记录op的shapes信息
            with_stack=True,  # 记录python层的代码stack
            with_flops=True,  # 估计矩阵乘和卷积的FLOPS（算力）
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')#dump json文件，没有这句话不会dump
        ) as prof:
            train(prof)
        expected_event_count = {
            # "+1" because the final iteration will enter __next__ but skip the loop body.
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#SGD.step": N,
            "Optimizer.zero_grad#SGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

        # Test on pickle/unpickle. Expect to work in multi-processing.
        optimizer = pickle.loads(pickle.dumps(optimizer))
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,  # 开启MLU profiler
            ],
        ) as prof:
            train()
        judge(expected_event_count, prof)

        # Test on customized optimizer.
        optimizer = CustomSGD(model.parameters(), lr=1e-4)
        with profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,  # 开启MLU profiler
            ],
        ) as prof:
            train()
        expected_event_count = {
            "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__": (N + 1),
            "Optimizer.step#CustomSGD.step": N,
            "Optimizer.zero_grad#CustomSGD.zero_grad": N,
        }
        judge(expected_event_count, prof)

    # @unittest.skip("not test")
    def test_profiler_autograd(self):
        x = torch.randn(10, 10).mlu()

        with _profile(use_mlu=True, use_kineto=True) as p:
            y = x * 2 + 4  # pylint: disable=W0612

        has_mul = False
        has_add = False

        for info in p.function_events:
            if info.name == "aten::add":
                has_add = True
            if info.name == "aten::mul":
                has_mul = True
        self.assertTrue(has_mul)
        self.assertTrue(has_add)

    # @unittest.skip("not test")
    @classmethod
    def test_profiler_autograd_aggregation(cls):
        total_time_s = 0
        with _profile(
            use_mlu=True, record_shapes=True, profile_memory=True, use_kineto=True
        ) as prof:
            start = time.time()
            x = torch.randn(
                (3, 6, 334, 334), requires_grad=True, dtype=torch.float32
            ).mlu()
            y = torch.pow(x, 3)
            y = y * 3 - 2
            y = y + 1.0
            y = y * 2.0
            y = y / 3
            y = torch.abs(y)
            # y = -y
            m = torch.nn.AvgPool2d((3, 3), stride=1)
            y = m(y)
            y = y / 1.3
            m = torch.nn.Softmax(dim=1)
            y = m(y)
            end = time.time()
            total_time_s += end - start
        # print(prof.table())
        print(prof.table(sort_by="self_cpu_time_total", row_limit=10, header="TEST"))
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=10
            )
        )

        # make it us which is profiler default
        total_time_us = total_time_s * 1000.0 * 1000.0
        print(
            "CPU time measurement python side overhead: {:.2f}%".format(
                (total_time_us / prof.self_cpu_time_total - 1.0) * 100.0
            )
        )

    @testinfo()
    def test_zero_element_input(self):
        with _profile(use_mlu=True, use_kineto=True) as prof:
            rnn = nn.LSTM(3, 3, 1, bias=True, bidirectional=True, proj_size=0)
            input = torch.randn(3, 0, 3)
            input_mlu = input.to("mlu")
            h0 = torch.randn(2, 0, 3).to("mlu")
            c0 = torch.randn(2, 0, 3).to("mlu")
            rnn_mlu = rnn.to("mlu")
            h0.requires_grad = True
            c0.requires_grad = True
            input_mlu.requires_grad = True
            input.requires_grad = True
            out_mlu, _ = rnn_mlu(input_mlu, (h0, c0))

            grad = torch.randn(out_mlu.shape).to("mlu")
            out_mlu.backward(grad)

        found_cnnlLSTMGatesForward = False
        found_cnnlLSTMGatesBackward = False

        for info in prof.function_events:
            if "cnnlLSTMGatesForward" in info.name:
                found_cnnlLSTMGatesForward = True
            if "cnnlLSTMGatesBackward" in info.name:
                found_cnnlLSTMGatesBackward = True

        self.assertTrue(found_cnnlLSTMGatesForward)
        self.assertTrue(found_cnnlLSTMGatesBackward)

    def _check_csv_output(self, dname, with_pre_kernel):
        self.assertTrue(os.path.exists(dname))
        json_file_num = 0
        json_file_names = []
        csv_dir_num = 0
        csv_file_nums = []
        timestamp_to_json_file = {}
        for json_file in os.listdir(dname):
            if os.path.isfile(os.path.join(dname, json_file)):
                parts = json_file.split(".")
                self.assertTrue(len(parts) > 4)
                self.assertTrue(
                    parts[-4].isdigit() and int(parts[-4]) > 0,
                    "Wrong tracing file name pattern",
                )
                self.assertEqual(parts[-3:], ["pt", "trace", "json"])
                json_file_num += 1
                json_file_names.append("-".join(parts[:2]))
                timestamp_to_json_file[parts[-4]] = json_file
            else:
                self.assertTrue(os.path.isdir(os.path.join(dname, "cambricon_output")))

        for csv_dir in os.listdir(os.path.join(dname, "cambricon_output")):
            self.assertTrue(csv_dir in json_file_names)
            csv_dir_num += 1
            csv_file_num = 0
            csv_file_list = []
            for csv_file in os.listdir(
                os.path.join(dname, "cambricon_output", csv_dir)
            ):
                self.assertTrue(csv_file.endswith("csv"))
                self.assertTrue(
                    self.check_csv_file(
                        os.path.join(dname, "cambricon_output", csv_dir, csv_file)
                    )
                )
                csv_file_num += 1
                csv_file_list.append(csv_file)
            csv_file_nums.append(csv_file_num)

            # assert kernel total time and kernel count are the same in json and csv files
            ts = csv_dir.split("-")[-1]
            self.assertTrue(ts in timestamp_to_json_file)
            json_file_name = timestamp_to_json_file.get(ts)
            (
                kernel_total_time_json,
                kernel_count_json,
            ) = get_kernel_total_info_from_json(os.path.join(dname, json_file_name))
            kernel_details_csv_file = "kernel_details.csv"
            self.assertTrue(kernel_details_csv_file in csv_file_list)
            (
                kernel_total_time_csv1,
                kernel_count_csv1,
                header_csv1,
            ) = get_kernel_total_info_from_kernel_details_csv(
                os.path.join(
                    dname, "cambricon_output", csv_dir, kernel_details_csv_file
                )
            )
            self.assertEqual(header_csv1[7], "Duration(us)")
            kernel_statistic_csv_file = "kernel_statistic.csv"
            self.assertTrue(kernel_statistic_csv_file in csv_file_list)
            (
                kernel_total_time_csv2,
                kernel_count_csv2,
                header_csv2,
            ) = get_kernel_total_info_from_kernel_statistic_csv(
                os.path.join(
                    dname, "cambricon_output", csv_dir, kernel_statistic_csv_file
                )
            )
            self.assertEqual(header_csv2[1], "Count")
            self.assertEqual(header_csv2[2], "Total Time(us)")
            op_kernel_statistic_csv_file = "op_kernel_statistic.csv"
            self.assertTrue(op_kernel_statistic_csv_file in csv_file_list)
            (
                kernel_total_time_csv3,
                kernel_count_csv3,
                header_csv3,
            ) = get_kernel_total_info_from_op_kernel_statistic_csv(
                os.path.join(
                    dname, "cambricon_output", csv_dir, op_kernel_statistic_csv_file
                )
            )
            self.assertEqual(header_csv3[2], "Count")
            self.assertEqual(header_csv3[3], "Total Time(us)")
            if with_pre_kernel:
                # json save all kernels, csv only save kernels of active step
                self.assertGreater(kernel_total_time_json, kernel_total_time_csv1)
                self.assertAlmostEqual(
                    kernel_total_time_csv1, kernel_total_time_csv2, places=2
                )
                self.assertAlmostEqual(
                    kernel_total_time_csv1, kernel_total_time_csv3, places=2
                )
                self.assertGreater(kernel_count_json, kernel_count_csv1)
                self.assertEqual(kernel_count_csv1, kernel_count_csv2)
                self.assertEqual(kernel_count_csv1, kernel_count_csv3)
            else:
                self.assertAlmostEqual(
                    kernel_total_time_json, kernel_total_time_csv1, places=2
                )
                self.assertAlmostEqual(
                    kernel_total_time_json, kernel_total_time_csv2, places=2
                )
                self.assertAlmostEqual(
                    kernel_total_time_json, kernel_total_time_csv3, places=2
                )
                self.assertEqual(kernel_count_json, kernel_count_csv1)
                self.assertEqual(kernel_count_json, kernel_count_csv2)
                self.assertEqual(kernel_count_json, kernel_count_csv3)

        self.assertEqual(json_file_num, csv_dir_num)
        self.assertTrue(all([num == csv_file_nums[0] for num in csv_file_nums]))

    @testinfo()
    def test_profiler_generate_csv_files(self):
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    self.payload(use_mlu=True)
                    p.step()

            self._check_csv_output(dname, with_pre_kernel=False)

    @testinfo()
    def test_profiler_result_method_with_different_time_granularity(self):
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU,
            ]
        )
        prof.start()
        self.payload(use_mlu=True)
        torch.mlu.synchronize()
        result = _disable_profiler()
        self.assertEqual(result.trace_start_us(), result.trace_start_ns() // 1000)
        for evt in result.events():
            self.assertEqual(evt.start_us(), evt.start_ns() // 1000)
            self.assertEqual(evt.duration_us(), evt.duration_ns() // 1000)

    @testinfo()
    def test_profiler_with_mlu_graph(self):
        g = torch.mlu.MLUGraph()
        x = torch.randn(100, 64, 32, 256, device="mlu")

        with torch.mlu.graph(g):
            y = x.clone()
            for i in range(10):
                y = torch.nn.functional.gelu(y)
        xm = torch.randn(100, 64, 32, 256, device="mlu")
        x.copy_(xm)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(3):
                    g.replay()
                    p.step()

            self._check_csv_output(dname, with_pre_kernel=True)
            found_tasktopo_invoke = False
            found_gelu_kernel = False
            gelu_kernel_num = 0
            for evt in p.events():
                if "cnTaskTopoEntityInvoke" in evt.name:
                    found_tasktopo_invoke = True
                if evt.device_type == DeviceType.PrivateUse1:
                    if "Gelu" in evt.name:
                        found_gelu_kernel = True
                        gelu_kernel_num += 1

            self.assertTrue(found_tasktopo_invoke)
            self.assertTrue(found_gelu_kernel)
            self.assertGreater(gelu_kernel_num, 1)

            found_tasktopo_external_op = False
            for json_file in os.listdir(dname):
                if os.path.isfile(os.path.join(dname, json_file)):
                    with io.open(os.path.join(dname, json_file), "r") as f:
                        events = json.load(f)["traceEvents"]
                        kernels = [
                            e
                            for e in events
                            if "cat" in e.keys() and e["cat"] == "kernel"
                        ]
                        for kernel in kernels:
                            if "tasktopo_external_op" in kernel["args"].keys():
                                found_tasktopo_external_op = True
            # If not enable TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP, Profiler will not write
            # "tasktopo_external_op" into json file.
            # The env enabled testcase in test_profiler_with_mlugraph.py
            self.assertFalse(found_tasktopo_external_op)

    @testinfo()
    def test_profiler_csv_CallTreeRoot_display(self):
        json_content = """
            [{
              "ph": "X", "cat": "mlu_runtime", "name": "cnInvokeKernel", "pid": 731020, "tid": 731020,
              "ts": 8896795.623, "dur": 14.667,
              "args": {
                "External id": 0,
                "cbid": 91, "correlation": 4625
              }
            },
            {
              "ph": "X", "cat": "mlu_runtime", "name": "cnnlMultiTensorScale", "pid": 731020, "tid": 731020,
              "ts": 8896779.555, "dur": 35.608,
              "args": {
                "External id": 0,
                "cbid": 313, "correlation": 4623
              }
            },
            {
              "ph": "X", "cat": "kernel", "name": "void MLUBlockKernel3StagePipelineMultiTensorScaleFast<float>(TensorListMetadata, float, int*)", "pid": 0, "tid": 1,
              "ts": 8896815.781, "dur": 23.600,
              "args": {
                "External id": 0,

                "queued": 0, "device": 0, "context": 1,
                "stream": 1, "correlation": 4625,
                "kernel type": "BLOCK",
                "dimx": 1, "dimy": 640, "dimz": 1,
                "tasktopo": 0,
                "tasktopo_node": 0
              }
            }]
        """
        with TemporaryDirectoryName() as dname:
            trace_json = json.loads(json_content)
            trace_json = {"schemaVersion": 1, "traceEvents": trace_json}
            trace_file_path = os.path.join(
                dname, "default_worker.default_span.pt.trace.json"
            )
            with open(trace_file_path, "w") as ofile:
                json.dump(trace_json, ofile, indent=4)

            self.assertTrue(os.path.exists(trace_file_path))

            from torch_mlu.profiler.analysis.api import analyze_data

            analyze_data(trace_file_path)

            kernel_details_csv_file_path = os.path.join(
                dname,
                "cambricon_output",
                "default_worker-default_span",
                "kernel_details.csv",
            )
            self.assertTrue(os.path.exists(kernel_details_csv_file_path))

            with open(kernel_details_csv_file_path, "r", newline="") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                self.assertTrue(header)
                self.assertTrue("Kernel Name" in header)
                self.assertTrue("Operator" in header)
                line1 = next(reader)
                self.assertTrue(line1)
                for i in range(len(header)):
                    if header[i] == "Kernel Name":
                        self.assertTrue(
                            line1[i]
                            == "MLUBlockKernel3StagePipelineMultiTensorScaleFast"
                        )
                    if header[i] == "Operator":
                        self.assertTrue(line1[i] == "cnnlMultiTensorScale")

    @testinfo()
    def test_profiler_generate_csv_files_with_record_shapes(self):
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                record_shapes=True,
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                self.payload(use_mlu=True)

            self.assertTrue(os.path.exists(dname))

            for csv_dir in os.listdir(os.path.join(dname, "cambricon_output")):
                csv_file_list = []
                for csv_file in os.listdir(
                    os.path.join(dname, "cambricon_output", csv_dir)
                ):
                    self.assertTrue(csv_file.endswith("csv"))
                    self.assertTrue(
                        self.check_csv_file(
                            os.path.join(dname, "cambricon_output", csv_dir, csv_file)
                        )
                    )
                    csv_file_list.append(csv_file)

                # check 'Input Shapes' and 'Input Type' in operator_details.csv
                operator_details_csv_file = "operator_details.csv"
                self.assertTrue(operator_details_csv_file in csv_file_list)
                with open(
                    os.path.join(
                        dname, "cambricon_output", csv_dir, operator_details_csv_file
                    ),
                    "r",
                    newline="",
                ) as csvfile1:
                    reader = csv.reader(csvfile1)
                    # ['Thread Id', 'Name', 'Input Shapes', 'Input Type', ... ]
                    header = next(reader)
                    self.assertEqual(header[1], "Name")
                    self.assertEqual(header[2], "Input Shapes")
                    self.assertEqual(header[3], "Input Type")
                    for row in reader:
                        if row[1] == "aten::mm":
                            self.assertEqual(row[2], "[[10, 10], [10, 10]]")
                            self.assertEqual(row[3], "['float', 'float']")
                        elif row[1] == "aten::add":
                            self.assertEqual(row[2], "[[10, 10], [10, 10], []]")
                            self.assertEqual(row[3], "['float', 'float', 'Scalar']")
                        else:
                            assert row[2] != "[]", "Input Shapes empty!"
                            assert row[3] != "", "Input Type empty!"

                # check 'Operator Input Shapes' and 'Operator Input Type' in kernel_details.csv
                kernel_details_csv_file = "kernel_details.csv"
                self.assertTrue(kernel_details_csv_file in csv_file_list)
                with open(
                    os.path.join(
                        dname, "cambricon_output", csv_dir, kernel_details_csv_file
                    ),
                    "r",
                    newline="",
                ) as csvfile2:
                    reader = csv.reader(csvfile2)
                    # ['Thread Id', 'Correlation Id', 'Kernel Name', 'Operator',
                    # 'Operator Input Shapes', 'Operator Input Type', ... ]
                    header = next(reader)
                    self.assertEqual(header[3], "Operator")
                    self.assertEqual(header[4], "Operator Input Shapes")
                    self.assertEqual(header[5], "Operator Input Type")
                    for row in reader:
                        if row[3] == "aten::mm":
                            self.assertEqual(row[4], "[[10, 10], [10, 10]]")
                            self.assertEqual(row[5], "['float', 'float']")
                        elif row[3] == "aten::add":
                            self.assertEqual(row[4], "[[10, 10], [10, 10], []]")
                            self.assertEqual(row[5], "['float', 'float', 'Scalar']")
                        else:
                            assert row[4] != "[]", "Operator Input Shapes empty!"
                            assert row[5] != "", "Operator Input Type empty!"


if __name__ == "__main__":
    unittest.main()
