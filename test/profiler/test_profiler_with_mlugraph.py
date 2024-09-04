from __future__ import print_function

import sys
import json
import csv
import os
import unittest
import logging
import threading
import random
import socket

import torch
import torch_mlu
import torch.nn as nn
import torch.nn.functional as F

from torch.profiler import profile
from torch.testing._internal.common_utils import (
    TemporaryFileName,
    TemporaryDirectoryName,
)
import torch.distributed as dist
import torch.multiprocessing as mp

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0411

logging.basicConfig(level=logging.DEBUG)

os.environ["TORCH_MLU_ENABLE_CATCHING_MLUGRAPH_OP"] = "1"


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        return x


class TestProfiler(TestCase):
    def assert_output(self, dname, expect_found, extra_op=[]):
        self.assertTrue(os.path.exists(dname))
        for json_file in os.listdir(dname):
            if os.path.isfile(os.path.join(dname, json_file)):
                with open(os.path.join(dname, json_file)) as f:
                    events = json.load(f)["traceEvents"]

                found_conv = False
                found_max_pool = False
                found_extra_op = [False] * len(extra_op)
                kernels = [
                    e for e in events if "cat" in e.keys() and e["cat"] == "kernel"
                ]
                for kernel in kernels:
                    args = kernel["args"]
                    if "tasktopo_external_op" in args.keys():
                        if "aten::_convolution" in args["tasktopo_external_op"]:
                            found_conv = True
                        elif "aten::max_pool" in args["tasktopo_external_op"]:
                            found_max_pool = True
                        for i, op in enumerate(extra_op):
                            if op in args["tasktopo_external_op"]:
                                found_extra_op[i] = True
                                break
                self.assertTrue(found_conv == expect_found)
                self.assertTrue(found_max_pool == expect_found)
                for res in found_extra_op:
                    self.assertTrue(res)
            else:
                self.assertTrue(os.path.isdir(os.path.join(dname, "cambricon_output")))
        for csv_dir in os.listdir(os.path.join(dname, "cambricon_output")):
            kernel_details_file = os.path.join(
                dname, "cambricon_output", csv_dir, "kernel_details.csv"
            )
            op_kernel_statistic_file = os.path.join(
                dname, "cambricon_output", csv_dir, "op_kernel_statistic.csv"
            )
            for csv_file in [kernel_details_file, op_kernel_statistic_file]:
                found_conv = False
                found_max_pool = False
                found_extra_op = [False] * len(extra_op)
                with open(csv_file, "r", newline="") as cf:
                    reader = csv.reader(cf)
                    header = next(reader)
                    self.assertTrue("Operator" in header)
                    op_idx = header.index("Operator")
                    for row in reader:
                        if "aten::_convolution" in row[op_idx]:
                            found_conv = True
                        elif "aten::max_pool" in row[op_idx]:
                            found_max_pool = True
                        for i, op in enumerate(extra_op):
                            if op in row[op_idx]:
                                found_extra_op[i] = True
                                break
                self.assertTrue(found_conv == expect_found)
                self.assertTrue(found_max_pool == expect_found)
                for res in found_extra_op:
                    self.assertTrue(res)

    @testinfo()
    def test_mlugraph_capture_in_profiler(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=3),
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                with torch.mlu.graph(g):
                    model(static_input)
                for _ in range(10):
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname, expect_found=True)

    @testinfo()
    def test_mlugraph_capture_outside_of_profiler(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile():
            with torch.mlu.graph(g):
                model(static_input)
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
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname, expect_found=True)

    @testinfo()
    def test_profiler_with_mlugraph_replay_only(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with torch.mlu.graph(g):
            model(static_input)
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
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname, expect_found=False)

    @testinfo()
    def test_sub_thread_capture_and_main_thread_replay(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()

        def graph_capture():
            with profile():
                with torch.mlu.graph(g):
                    model(static_input)

        graph_capture_thread = threading.Thread(target=graph_capture)
        graph_capture_thread.start()
        graph_capture_thread.join()
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
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname, expect_found=True)

    @testinfo()
    def test_main_thread_capture_and_sub_thread_replay(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile():
            with torch.mlu.graph(g):
                model(static_input)

        def replay_and_assert():
            with TemporaryDirectoryName() as dname:
                with profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MLU,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=1, warmup=1, active=2, repeat=3
                    ),
                    on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
                ) as p:
                    for _ in range(18):
                        x = torch.randn(1, 3, 128, 128).mlu()
                        static_input.copy_(x)
                        g.replay()
                        p.step()
                self.assert_output(dname, expect_found=True)

        replay_thread = threading.Thread(target=replay_and_assert)
        replay_thread.start()
        replay_thread.join()

    @testinfo()
    def test_profiler_with_multi_mlugraph(self):
        g1 = torch.mlu.MLUGraph()
        static_input_g1 = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile():
            with torch.mlu.graph(g1):
                model(static_input_g1)

        g2 = torch.mlu.MLUGraph()
        static_input_g2 = torch.randn(1, 64, 32, 256, device="mlu")
        with profile():
            with torch.mlu.graph(g2):
                for i in range(5):
                    torch.nn.functional.gelu(static_input_g2)

        # Custom OP
        g3 = torch.mlu.MLUGraph()
        static_input_g3 = torch.randn([54, 7]).mlu()
        static_other_g3 = torch.randn([50, 7]).mlu()
        with profile():
            with torch.mlu.graph(g3):
                torch.ops.torch_mlu.boxes_overlap_bev(static_input_g3, static_other_g3)

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
                    x1 = torch.randn(1, 3, 128, 128).mlu()
                    static_input_g1.copy_(x1)
                    x2 = torch.randn(1, 64, 32, 256).mlu()
                    static_input_g2.copy_(x2)
                    x3_input = torch.randn([54, 7]).mlu()
                    x3_other = torch.randn([50, 7]).mlu()
                    static_input_g3.copy_(x3_input)
                    static_other_g3.copy_(x3_other)
                    g1.replay()
                    g2.replay()
                    g3.replay()
                    p.step()
            self.assert_output(
                dname,
                expect_found=True,
                extra_op=["aten::gelu", "torch_mlu::boxes_overlap_bev"],
            )

    @staticmethod
    def setup(rand, world_size, master_port):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("cncl", rank=rand, world_size=world_size)

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

    @staticmethod
    def run_all_reduce_in_mlugraph(rank, world_size, master_port, dname):
        rank = rank
        TestProfiler.setup(rank, world_size, master_port)
        device = torch.device(f"mlu:{rank}")
        torch.mlu.set_device(device)
        a = torch.randn(10, 3, device="mlu")
        g = torch.mlu.MLUGraph()
        with torch.profiler.profile():
            with torch.mlu.graph(g):
                dist.all_reduce(a)
        with torch.profiler.profile(
            on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname)
        ) as p:
            x = torch.randn(10, 3, device="mlu")
            a.copy_(x)
            g.replay()
            dist.barrier()
        TestProfiler.cleanup()

    @testinfo()
    @unittest.skipIf(torch.mlu.device_count() < 2, "Test requres 2 MLU devices")
    @unittest.skipIf(
        "590" not in torch.mlu.get_device_name(), "Test only for MLU590 series"
    )
    def test_profiler_with_allreduce_in_mlugraph(self):
        def find_free_port():
            while True:
                port = random.randint(1024, 65535)
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    if s.connect_ex(("localhost", port)) != 0:
                        return port

        world_size = 2
        master_port = find_free_port()
        with TemporaryDirectoryName() as dname:
            mp.spawn(
                TestProfiler.run_all_reduce_in_mlugraph,
                args=(world_size, master_port, dname),
                nprocs=world_size,
                join=True,
            )
            self.assert_output(dname, expect_found=False, extra_op=["c10d::allreduce_"])


if __name__ == "__main__":
    unittest.main()
