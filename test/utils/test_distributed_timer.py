import os
import sys
import logging
import unittest
import re

import torch
import torch_mlu
import torch.distributed as dist
import torch.multiprocessing as mp

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    OutputGrabber,
)  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29501"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.
os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"


class TwoLinLayerNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(10, 10, bias=False)
        self.b = torch.nn.Linear(10, 1, bias=False)

    def forward(self, x):
        a = self.a(x)
        b = self.b(x)
        return (a, b)


def worker(rank):
    dist.init_process_group("cncl", rank=rank, world_size=2)
    torch.mlu.set_device(rank)
    print("init model")
    model = TwoLinLayerNet().mlu()
    print("init ddp")
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    inp = torch.randn(10, 10).mlu()
    print("train")

    for _ in range(20):
        output = ddp_model(inp)
        loss = output[0] + output[1]
        loss.sum().backward()

    dist.barrier()


class TestDistributedTimer(TestCase):
    def checkout_time_greater_than_zero(self, input, pattern):
        input_list = input.split("\n")
        for input_string in input_list:
            match = re.search(pattern, input_string)
            if match:
                self.assertTrue(int(match.group(1)) > 0)

    @unittest.skip("not test, see PYTORCH-11921")
    @testinfo()
    def test_distributed_timer(self):
        out = OutputGrabber(sys.stderr)
        out.start()
        mp.spawn(worker, nprocs=2, args=())
        out.stop()
        pattern_list = [
            r"Avg forward compute time: (\d+)",
            r"Avg backward compute time: (\d+)",
            r"Avg backward comm. time: (\d+)",
            r"Avg backward comm/comp overlap time: (\d+)",
        ]
        for pattern in pattern_list:
            self.checkout_time_greater_than_zero(out.capturedtext, pattern)
        # print(f"out captureed {out.capturedtext}")


if __name__ == "__main__":
    unittest.main()
