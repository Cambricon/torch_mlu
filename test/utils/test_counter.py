import sys
import logging
import os
import io
import contextlib
from importlib import reload
import shutil
import unittest
import multiprocessing as mp
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.testing._internal.common_utils import find_free_port
import torch.distributed as dist
import torch_mlu
from torch_mlu.utils.counter import _check_gencase

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

INIT_METHOD = "tcp://127.0.0.5:" + str(find_free_port())
TIMEOUT = 200


def spawn_processes(world_size, func):
    processes = []
    # start task
    for rank in range(world_size):
        name = "process " + str(rank)
        process = mp.Process(target=func, name=name, args=(rank, world_size))
        process.start()
        processes.append(process)
    # wait task completion
    for rank, process in enumerate(processes):
        process.join(TIMEOUT)
        if process.is_alive():
            print(f"Timeout waiting for rank {rank} to terminate")
            process.terminate()


def dataloader_test_helper(rank=0):
    inps = torch.arange(30 * 5, dtype=torch.float32).view(30, 5).to(f"mlu:{rank}")
    tgts = torch.arange(30 * 5, dtype=torch.float32).view(30, 5).to(f"mlu:{rank}")
    dataset = TensorDataset(inps, tgts)

    loader = DataLoader(dataset, batch_size=2)
    for _, sample in enumerate(loader):
        sample[0] + sample[1]  # pylint: disable=W0104


class TestTorchMLUCounter(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_counter_file(self):
        gencase_dir = "./gen_case"
        for level in ["L1", "L2", "L3"]:
            os.environ["TORCH_MLU_COUNTER_LEVEL"] = level
            reload(torch_mlu.utils.counter)
            _check_gencase()
            if os.path.exists(gencase_dir):
                shutil.rmtree(gencase_dir)
                print("gencase_file removed during test")
            dataloader_test_helper()
            generated_files_dir = gencase_dir + "/op_tensor"
            if level != "L3":
                self.assertTrue(os.path.exists(generated_files_dir))

    # @unittest.skip("not test")
    @testinfo()
    def test_counter_number(self):
        range_list = [[2, 3], [4, 6], [0, 16], [13, 16], [12, 17], [0, 2]]
        for counter_range in range_list:
            os.environ["TORCH_MLU_COUNTER_START"] = str(counter_range[0])
            os.environ["TORCH_MLU_COUNTER_END"] = str(counter_range[1])
            reload(torch_mlu.utils.counter)
            _check_gencase()
            reference = min(counter_range[1] - 1, 15)
            with io.StringIO() as f:
                with contextlib.redirect_stdout(f):
                    dataloader_test_helper()
                self.assertTrue(str(reference) in f.getvalue().splitlines()[-1])

    # @unittest.skip("not test")
    @testinfo()
    def test_gencase_warnings(self):
        warining_msg_list = [
            "TORCH_MLU_COUNTER_START and TORCH_MLU_COUNTER_END should be integers",
            "TORCH_MLU_COUNTER_START and TORCH_MLU_COUNTER_END should"
            " all be greater than 0",
            "TORCH_MLU_COUNTER_END should be greater than TORCH_MLU_COUNTER_START",
            "TORCH_MLU_COUNTER_START, TORCH_MLU_COUNTER_END, TORCH_MLU_COUNTER_LEVEL should be"
            " all set to enable torch_mlu counter",
            "TORCH_MLU_COUNTER enable failed, TORCH_MLU_COUNTER_LEVEL only support"
            " L1, L2, L3, but got L0",
            "TORCH_MLU_COUNTER enable failed",
        ]
        warning_value_list = ["1.1", "-1", "3", None, "L0", "1"]
        for warning_msg, warning_value in zip(warining_msg_list, warning_value_list):
            if warning_value == "L0":
                os.environ["TORCH_MLU_COUNTER_START"] = "1"
                os.environ["TORCH_MLU_COUNTER_LEVEL"] = warning_value
            elif warning_value is None:
                del os.environ["TORCH_MLU_COUNTER_START"]
            elif warning_value == "1":
                os.environ["CNNL_GEN_CASE"] = "1"
            else:
                os.environ["TORCH_MLU_COUNTER_START"] = warning_value
            with self.assertWarnsRegex(UserWarning, warning_msg):
                reload(torch_mlu.utils.counter)
                _check_gencase()
                dataloader_test_helper()

    # TODO(): There is a problem with cnnl under multi-process writing
    @unittest.skip("not test")
    @testinfo()
    # @unittest.skipIf(torch.mlu.device_count() < 2, "need at least 2 MLU")
    def test_counter_dist(self):
        def func_task(rank, world_size):
            dist.init_process_group(
                backend="gloo",
                init_method=INIT_METHOD,
                rank=rank,
                world_size=world_size,
            )
            reload(torch_mlu.utils.counter)
            _check_gencase()
            dataloader_test_helper(rank=rank)
            dist.barrier()

        spawn_processes(2, func_task)
        gencase_dir = "./gen_case"
        generated_files_dir = gencase_dir + "/op_tensor"
        self.assertTrue(os.path.exists(generated_files_dir))
        self.assertEqual(len(os.listdir(generated_files_dir)), 2)

    def setUp(self):
        super(TestTorchMLUCounter, self).setUp()
        os.environ["TORCH_MLU_COUNTER_START"] = "1"
        os.environ["TORCH_MLU_COUNTER_END"] = "2"
        os.environ["TORCH_MLU_COUNTER_LEVEL"] = "L1"

    def tearDown(self):
        super(TestTorchMLUCounter, self).tearDown()
        gencase_dir = "./gen_case"
        if os.path.exists(gencase_dir):
            shutil.rmtree(gencase_dir)
            print("gencase_file removed during teardown")


if __name__ == "__main__":
    unittest.main()
