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


class TestProfilerWithShapes(TestCase):
    def assert_output(self, dname, record_shapes=True):
        for csv_dir in os.listdir(os.path.join(dname, "cambricon_output")):
            operator_details_csv_file = os.path.join(
                dname, "cambricon_output", csv_dir, "operator_details.csv"
            )
            kernel_details_csv_file = os.path.join(
                dname, "cambricon_output", csv_dir, "kernel_details.csv"
            )
            # assert 'Input Shapes' and 'Input Type' are not empty in operator_details.csv
            self.assertTrue(os.path.exists(operator_details_csv_file))
            with open(operator_details_csv_file, "r", newline="") as csvfile1:
                reader = csv.reader(csvfile1)
                # ['Thread Id', 'Name', 'Input Shapes', 'Input Type', ... ]
                header = next(reader)
                self.assertEqual(header[1], "Name")
                self.assertEqual(header[2], "Input Shapes")
                self.assertEqual(header[3], "Input Type")
                for row in reader:
                    if not record_shapes:
                        assert row[2] == "[]", "Input Shapes not empty!"
                        assert row[3] == "", "Input Type not empty!"
                        continue
                    if row[1] == "aten::_convolution":
                        self.assertEqual(
                            row[2],
                            "[[1, 3, 128, 128], [16, 3, 3, 3], [16], [], [], [], [], [], [], [], [], [], []]",
                        )
                        self.assertEqual(
                            row[3],
                            "['float', 'float', 'float', 'ScalarList', 'ScalarList', 'ScalarList', 'Scalar', 'ScalarList', 'Scalar', 'Scalar', 'Scalar', 'Scalar', 'Scalar']",
                        )
                    else:
                        assert row[2] != "[]", "Input Shapes empty!"
                        assert row[3] != "", "Input Type empty!"

            # assert 'Operator Input Shapes' and 'Operator Input Type' are not empty in kernel_details.csv
            self.assertTrue(os.path.exists(kernel_details_csv_file))
            with open(kernel_details_csv_file, "r", newline="") as csvfile2:
                reader = csv.reader(csvfile2)
                # ['Thread Id', 'Correlation Id', 'Kernel Name', 'Operator',
                # 'Operator Input Shapes', 'Operator Input Type', ... ]
                header = next(reader)
                self.assertEqual(header[3], "Operator")
                self.assertEqual(header[4], "Operator Input Shapes")
                self.assertEqual(header[5], "Operator Input Type")
                for row in reader:
                    if not record_shapes:
                        assert row[4] == "[]", "Operator Input Shapes not empty!"
                        assert row[5] == "", "Operator Input Type not empty!"
                        continue
                    if row[3] == "aten::_convolution":
                        self.assertEqual(
                            row[4],
                            "[[1, 3, 128, 128], [16, 3, 3, 3], [16], [], [], [], [], [], [], [], [], [], []]",
                        )
                        self.assertEqual(
                            row[5],
                            "['float', 'float', 'float', 'ScalarList', 'ScalarList', 'ScalarList', 'Scalar', 'ScalarList', 'Scalar', 'Scalar', 'Scalar', 'Scalar', 'Scalar']",
                        )
                    elif row[3] == "c10d::allreduce_":
                        self.assertEqual(row[4], "[[], [], [], [], []]")
                        self.assertEqual(row[5], "['TensorList', '', '', '', 'Scalar']")
                    elif row[3] == "aten::gelu":
                        self.assertEqual(row[4], "[[1, 64, 32, 256], []]")
                        self.assertEqual(row[5], "['float', '']")
                    elif row[3] == "torch_mlu::boxes_overlap_bev":
                        self.assertEqual(row[4], "[[54, 7], [50, 7]]")
                        self.assertEqual(row[5], "['float', 'float']")
                    else:
                        assert row[4] != "[]", "Operator Input Shapes empty!"
                        assert row[5] != "", "Operator Input Type empty!"

    @testinfo()
    def test_mlugraph_capture_in_profiler_without_record_shapes(self):
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
                record_shapes=False,
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
            self.assert_output(dname, record_shapes=False)

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
                record_shapes=True,
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
            self.assert_output(dname)

    @testinfo()
    def test_mlugraph_capture_outside_of_profiler(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile(record_shapes=True):
            with torch.mlu.graph(g):
                model(static_input)
        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                record_shapes=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname)

    @testinfo()
    def test_sub_thread_capture_and_main_thread_replay(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()

        def graph_capture():
            with profile(record_shapes=True):
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
                record_shapes=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=3),
                on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
            ) as p:
                for _ in range(18):
                    x = torch.randn(1, 3, 128, 128).mlu()
                    static_input.copy_(x)
                    g.replay()
                    p.step()
            self.assert_output(dname)

    @testinfo()
    def test_main_thread_capture_and_sub_thread_replay(self):
        g = torch.mlu.MLUGraph()
        static_input = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile(record_shapes=True):
            with torch.mlu.graph(g):
                model(static_input)

        def replay_and_assert():
            with TemporaryDirectoryName() as dname:
                with profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.MLU,
                    ],
                    record_shapes=True,
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
                self.assert_output(dname)

        replay_thread = threading.Thread(target=replay_and_assert)
        replay_thread.start()
        replay_thread.join()

    @testinfo()
    def test_profiler_with_multi_mlugraph(self):
        g1 = torch.mlu.MLUGraph()
        static_input_g1 = torch.randn(1, 3, 128, 128).mlu()
        model = ConvNet()
        model = model.mlu()
        with profile(record_shapes=True):
            with torch.mlu.graph(g1):
                model(static_input_g1)

        g2 = torch.mlu.MLUGraph()
        static_input_g2 = torch.randn(1, 64, 32, 256, device="mlu")
        with profile(record_shapes=True):
            with torch.mlu.graph(g2):
                for i in range(5):
                    torch.nn.functional.gelu(static_input_g2)

        # Custom OP
        g3 = torch.mlu.MLUGraph()
        static_input_g3 = torch.randn([54, 7]).mlu()
        static_other_g3 = torch.randn([50, 7]).mlu()
        with profile(record_shapes=True):
            with torch.mlu.graph(g3):
                torch.ops.torch_mlu.boxes_overlap_bev(static_input_g3, static_other_g3)

        with TemporaryDirectoryName() as dname:
            with profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.MLU,
                ],
                record_shapes=True,
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
            self.assert_output(dname)

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
        TestProfilerWithShapes.setup(rank, world_size, master_port)
        device = torch.device(f"mlu:{rank}")
        torch.mlu.set_device(device)
        a = torch.randn(1, 3, 128, 128, device="mlu")
        g = torch.mlu.MLUGraph()
        with torch.profiler.profile(record_shapes=True):
            with torch.mlu.graph(g):
                dist.all_reduce(a)
        with torch.profiler.profile(
            record_shapes=True,
            on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(dname),
        ) as p:
            x = torch.randn(1, 3, 128, 128, device="mlu")
            a.copy_(x)
            g.replay()
            dist.barrier()
        TestProfilerWithShapes.cleanup()

    @testinfo()
    @unittest.skipIf(torch.mlu.device_count() < 2, "Test requres 2 MLU devices")
    @unittest.skipIf(
        "M9" not in torch.mlu.get_device_name(), "Test only for specific cards"
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
                TestProfilerWithShapes.run_all_reduce_in_mlugraph,
                args=(world_size, master_port, dname),
                nprocs=world_size,
                join=True,
            )
            self.assert_output(dname)


if __name__ == "__main__":
    unittest.main()
