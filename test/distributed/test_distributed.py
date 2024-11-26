from __future__ import absolute_import, division, print_function, unicode_literals

# pylint: disable=C0413,C0411,C0302
import copy
import os
import sys
from sys import path
from os.path import dirname
from itertools import product
import time
import unittest
import math
from functools import reduce, wraps
import distutils.dir_util
from argparse import ArgumentParser
from multiprocessing.managers import BaseManager
from queue import Queue
import random as rd
from contextlib import contextmanager
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import socket
from contextlib import closing

import fcntl

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    cleanup_temp_dir,
    initialize_temp_directories,
    with_dist_debug_levels,
)

from torch.testing._internal.common_utils import run_tests, FILE_SCHEMA

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

path.append(dirname(path[0]))
from common_utils import (
    TestCase,
    read_card_info,
    OutputGrabber,
    TEST_LARGETENSOR,
    largeTensorTest,
)

TEST_BFLOAT16 = read_card_info()
INIT_METHOD = os.getenv("INIT_METHOD", "env://")
DEFAULT_TIMEOUT = 300
CUSTOMIZED_TIMEOUT = {
    "test_distributedDataParallel": 200,
    "test_pressure": 200,
    "test_barrier": 300,
}
SKIP_IF_BACKEND_UNAVAILABLE = 78
cwd = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(cwd, "tmp")


def find_free_port():
    """
    Finds an available port and returns that port number.

    NOTE: If this function is being used to allocate a port to Store (or
    indirectly via init_process_group or init_rpc), it should be used
    in conjuction with the `retry_on_connect_failures` decorator as there is a potential
    race condition where the allocated port may become unavailable before it can be used
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", 0))
        _, port = sock.getsockname()
        return port


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(description="torch_mlu distributed training unittest")

    # Optional arguments for the launch helper
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="The number of nodes participate in testing, "
        "this is set ot 1 by default.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The rank of the node for multi-node testing, "
        "this is set to 0 by default.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=4,
        help="The number of processes to launch on each node, "
        "for multi-node testing, this is set to 4 by default.",
    )
    parser.add_argument(
        "--connects",
        default=rd.randint(-1, 4),
        type=int,
        choices=range(-1, 4),
        help="We support testing for different technologies of "
        "connection. Different techs have different priority "
        "levels. In this script, MLU‑Link > P2P > SHM > SOCKET. "
        "when input is -1, no cncl environment will be set; "
        "input is 0, all techs can be used; input is 1, only "
        "P2P, SHM and SOCKET can be used, MLU‑Link is prohibited; "
        "2: SHM, SOCKET; 3: SOCKET. By default, every techs "
        "have chances to be tested. Note: When here are multi "
        "node, connects would be set to -1 forcibly, please "
        "make sure different node use the same cncl environments "
        "according to cncl user guide doc.",
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.10",
        type=str,
        help="Master node (rank 0)'s address, should be either "
        "the IP address or the hostname of node 0, for "
        "single node multi-proc training, the "
        "--master_addr can simply be 127.0.0.1, now this is "
        "set to 127.0.0.10 by default.",
    )
    parser.add_argument(
        "--master_port",
        default=find_free_port(),
        type=int,
        help="Master node (rank 0)'s free port that needs to "
        "be used for communciation during distributed "
        "training, now this is set to 20000 by default. "
        "In addition, we also use (master_port + 10) for sync.",
    )
    parser.add_argument(
        "--delay_time",
        default=0,
        type=int,
        help="The communication time may be different between "
        "different environment. So we provide a choice for "
        "user to add additional delay time for all test case.",
    )

    parser.add_argument("unittest_args", nargs="*")

    return parser.parse_args()


def get_reduce_module():
    cpp_source = """
#include "framework/distributed/process_group_cncl.hpp"
#include "framework/core/MLUStream.h"
#include "aten/utils/tensor_util.h"

at::Tensor reduceSum(at::Tensor& input, int64_t cnclComm) {
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float,
              "dtype of input must be Float32.");
  cnclComm_t cncl_comm = reinterpret_cast<cnclComm_t>(cnclComm);
  const auto device = input.device();
  auto input_impl = torch_mlu::getMluTensorImpl(input);
  void* input_ptr = input_impl->mlu_data_ptr();
  auto stream = torch_mlu::getCurrentMLUStream(device.index());
  cnclAllReduce(input_ptr, input_ptr, input.numel(), cnclFloat32, cnclSum, cncl_comm, stream);
  return input;
}
    """
    from torch_mlu.utils.cpp_extension import include_paths, library_paths

    # currently, not support torch_mlu.utils.cpp_extension.load_inline.
    extra_ldflags = ["-ltorch_mlu"]
    for path in library_paths():
        extra_ldflags.append("-L" + path)
    module = torch.utils.cpp_extension.load_inline(
        name="process_group_extension",
        cpp_sources=cpp_source,
        extra_include_paths=include_paths(),
        extra_ldflags=extra_ldflags,
        functions=["reduceSum"],
    )
    return module


def _build_tensor(size, value=None, dtype=torch.float, device_id=None):
    if value is None:
        value = size
    if device_id is None:
        return torch.empty(size, size, size, dtype=dtype).fill_(value)
    else:
        return torch.empty(size, size, size, dtype=dtype).fill_(value).mlu(device_id)


def get_timeout(test_id):
    test_name = test_id.split(".")[-1]
    if test_name in CUSTOMIZED_TIMEOUT:
        return CUSTOMIZED_TIMEOUT[test_name]
    else:
        return DEFAULT_TIMEOUT


class QueueManager(BaseManager):
    pass


# flg: flg==True means force run OP on CPU, to avoid MLU caculations.
class Linear_mlu(nn.Linear):  # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            if self.bias is not None:
                bias_cpu = self.bias.cpu()
            else:
                bias_cpu = None
            return F.linear(input_.cpu(), self.weight.cpu(), bias_cpu)
        else:
            return F.linear(input_, self.weight, self.bias)


class Conv2d_mlu(nn.Conv2d):  # pylint: disable=W0223
    def forward(self, input_, flg):
        if flg:
            return self._conv_forward(input_.cpu(), self.weight.cpu(), None)
        else:
            return self._conv_forward(input_, self.weight, None)


class _FC2(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(_FC2, self).__init__()  # pylint: disable=R1725
        self.fc = Linear_mlu(10, 12, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x, flg):
        x = self.fc(x, flg)
        return x


class Net(nn.Module):  # pylint: disable=W0223
    def __init__(self):
        super(Net, self).__init__()  # pylint: disable=R1725
        self.fc1 = Linear_mlu(2, 10, bias=False)
        self.fc2 = _FC2()
        self.conv = Conv2d_mlu(2, 6, 2, bias=False)
        self.fc3 = Linear_mlu(12, 4, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.Tensor([2, 2]).long(), requires_grad=False
        )

    def forward(self, x, flg):
        x = self.relu(self.fc1(x, flg))
        x = self.relu(self.fc2(x, flg))
        x = self.conv(x.view(-1, 2, 3, 2), flg).view(-1, 12)
        x = self.fc3(x, flg)
        return F.softmax(x, dim=1)


class BatchNormNet(nn.Module):  # pylint: disable=W0223
    def __init__(self, affine=True):
        super(BatchNormNet, self).__init__()  # pylint: disable=R1725
        self.fc1 = Linear_mlu(2, 40, bias=False)
        self.bn = nn.BatchNorm1d(4, affine=affine)
        self.fc2 = Linear_mlu(40, 4, bias=False)

    def forward(self, x, flg):
        x = torch.reshape(self.fc1(x, flg), (-1, 4, 10))
        x = self.bn(x.to("mlu")).cpu()
        x = torch.reshape(x, (-1, 40))
        x = self.fc2(x, flg)
        return F.softmax(x, dim=1)


class OnlyBatchNormNet(nn.Module):  # pylint: disable=W0223
    def __init__(self, module):
        super(OnlyBatchNormNet, self).__init__()  # pylint: disable=R1725
        self.bn = module

    def forward(self, x, flg):  # pylint: disable=W0613
        x = self.bn(x.to("mlu")).cpu()
        return x


class Foo:
    def __init__(self, x):
        # Can be tensor or int
        self.x = x

    def __eq__(self, other):
        def eq(value, other):
            if isinstance(value, torch.Tensor):
                return torch.equal(value, other)
            return value == other

        for attr, value in self.__dict__.items():
            other_value = other.__dict__[attr]
            if not eq(value, other_value):
                return False
        return True


class _DistTestBase(object):  # pylint: disable=R0205, R0904
    args = None
    rank = 0
    world_size = 0

    def _barrier(self, *args, **kwargs):
        Barrier.sync(*args, **kwargs)

    @classmethod
    def _init_global_test(cls):
        group = [i for i in range(0, dist.get_world_size())]  # pylint: disable=R1721
        rank = dist.get_rank()
        return (group, rank)

    def _test_broadcast_helper(self, group, rank):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        type_info = [
            ("torch.FloatTensor", -1e-10, True),
            ("torch.HalfTensor", -0.1, True),
            ("torch.CharTensor", -2, True),
            ("torch.ByteTensor", 129, True),
            ("torch.IntTensor", -1e5, True),
            ("torch.LongTensor", 1e5, True),
            ("torch.DoubleTensor", -1e-10, True),
        ]
        if TEST_BFLOAT16:
            type_info.append(("torch.BFloat16Tensor", -1e-10, True))
        for ttype, value, is_test in type_info:
            if not is_test:
                continue
            for src in group:
                expected_tensor = torch.tensor([]).mlu()
                expected_tensor = _build_tensor(src + 1, value).type(ttype).mlu()
                if rank == src:
                    dist.broadcast(expected_tensor, src)
                else:
                    tensor = _build_tensor(src + 1, -1).type(ttype).mlu()
                    dist.broadcast(tensor, src)
                    self.assertEqual(tensor.size(), expected_tensor.size())
                    self.assertTrue(
                        tensor.type(torch.float)
                        .cpu()
                        .eq(expected_tensor.type(torch.float).cpu())
                        .min()
                        .item()
                    )

    # @unittest.skip("not test")
    def test_broadcast(self):
        group, rank = self._init_global_test()
        self._test_broadcast_helper(group, rank)

    # @unittest.skip("not test")
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE`"
    )
    @largeTensorTest("29GB", device="mlu")
    def test_broadcast_large(self):
        group, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        value = 1e5
        size = 1073741824
        for src in group:
            expected_tensor = torch.empty(size, dtype=torch.long).fill_(value).mlu()
            if rank == src:
                dist.broadcast(expected_tensor, src)
            else:
                tensor = torch.empty(size, dtype=torch.long).fill_(-1).mlu()
                dist.broadcast(tensor, src)
                self.assertEqual(tensor.size(), expected_tensor.size())
                self.assertTrue(
                    tensor.type(torch.float)
                    .cpu()
                    .eq(expected_tensor.type(torch.float).cpu())
                    .min()
                    .item()
                )

    # @unittest.skip("not test")
    def test_broadcast_object_list(self):
        # Case where rank != MLU device.
        next_rank = (self.rank + 1) % torch.mlu.device_count()
        torch.mlu.set_device(next_rank)

        torch.manual_seed(0)
        f = Foo(10)
        f.bar = 1
        foo_cpu_tensor = Foo(torch.randn(3, 3))
        foo_mlu_tensor = Foo(torch.randn(3, 3).mlu(0))
        COLLECTIVES_OBJECT_TEST_LIST = [
            {"key1": 3, "key2": 4, "key3": {"nested": True}},
            f,
            foo_cpu_tensor,
            foo_mlu_tensor,
            "foo",
            [1, 2, True, "string", [4, 5, "nested"]],
        ]

        src_rank = 0

        objects = (
            COLLECTIVES_OBJECT_TEST_LIST
            if self.rank == src_rank
            else [None for _ in COLLECTIVES_OBJECT_TEST_LIST]
        )

        # Single object test
        single_obj_list = [objects[0]]
        if self.rank != src_rank:
            self.assertNotEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])
        dist.broadcast_object_list(single_obj_list, src=0)
        self.assertEqual(single_obj_list[0], COLLECTIVES_OBJECT_TEST_LIST[0])

        # Multiple input objects test
        if self.rank != src_rank:
            self.assertNotEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)

        dist.broadcast_object_list(objects, src=0)
        # Test mlu tensor broadcast successfully
        self.assertTrue(objects[3].x.device.type == "mlu")
        self.assertEqual(objects, COLLECTIVES_OBJECT_TEST_LIST)

    def _test_async_helper(
        self, group, rank, op, master_value, worker_value, expected_value
    ):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for src in group:
            if rank == src:
                tensor = self.to_device(_build_tensor(src + 1, master_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertEqual(
                    tensor.cpu(),
                    _build_tensor(src + 1, expected_value),
                    atol=3e-3,
                    rtol=0,
                )
            else:
                tensor = self.to_device(_build_tensor(src + 1, worker_value))
                work = dist.all_reduce(tensor, op, async_op=True)
                work.wait()
                self.assertEqual(
                    tensor.cpu(),
                    _build_tensor(src + 1, expected_value),
                    atol=3e-3,
                    rtol=0,
                )
            self.assertTrue(work.is_completed())
            self.assertTrue(work.is_success())

    # @unittest.skip("not test")
    def test_async(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = torch.randn(1).item()
        b = torch.randn(1).item()
        self._test_async_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    def _test_reduce_helper(
        self, group, rank, op, master_value, worker_value, expected_value
    ):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        dtype_list = [
            ("torch.FloatTensor", True),
            ("torch.HalfTensor", True),
            ("torch.IntTensor", True),
            ("torch.LongTensor", True),
        ]
        if TEST_BFLOAT16:
            dtype_list.append(("torch.BFloat16Tensor", True))
        list_list = [dtype_list, group]

        for dtype_tuple, src in product(*list_list):
            if not dtype_tuple[1]:
                continue
            ttype = dtype_tuple[0]
            if rank == src:
                tensor = self.to_device(
                    _build_tensor(src + 1, master_value).type(ttype)
                )
                dist.reduce(tensor, src, op)
                self.assertLess(
                    (tensor.float() - _build_tensor(src + 1, expected_value).to("mlu"))
                    .abs()
                    .cpu()
                    .max()
                    .item(),
                    3e-3,
                )
            else:
                tensor = self.to_device(
                    _build_tensor(src + 1, worker_value).type(ttype)
                )
                dist.reduce(tensor, src, op)

    # @unittest.skip("not test")
    def test_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = 10
        b = 2
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    # @unittest.skip("not test")
    def test_custom_reduce_sum(self):
        _, rank = self._init_global_test()
        world_size = dist.get_world_size()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        module = get_reduce_module()
        pg = _get_default_group()
        cncl_comm = pg._get_backend(torch.device("mlu")).get_cncl_comm(rank)
        input = _build_tensor(5, 1, device_id=device_id)
        out = module.reduceSum(input, cncl_comm)
        expected_tensor = torch.ones(5, 5, 5) * world_size
        self.assertEqual(out.cpu(), expected_tensor)

    # @unittest.skip("not test")
    def test_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    # @unittest.skip("not test")
    def test_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    # @unittest.skip("not test")
    def test_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    # @unittest.skip("not test")
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE`"
    )
    @largeTensorTest("21GB", device="mlu")
    def test_reduce_sum_large(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        master_value = torch.randn(1).item()
        worker_value = torch.randn(1).item()
        expected_value = master_value + worker_value * (len(group) - 1)
        op = dist.ReduceOp.SUM
        size = 1073741824
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for src in group:
            expected_tensor = (
                torch.empty(size, dtype=torch.float).fill_(expected_value).mlu()
            )
            if rank == src:
                tensor = torch.empty(size, dtype=torch.float).fill_(master_value).mlu()
                dist.reduce(tensor, src, op)
                self.assertLess(
                    (tensor.float() - expected_tensor).abs().cpu().max().item(), 3e-3
                )
            else:
                tensor = torch.empty(size, dtype=torch.float).fill_(worker_value).mlu()
                dist.reduce(tensor, src, op)

    def _test_all_reduce_helper(
        self, group, rank, op, master_value, worker_value, expected_value
    ):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        dtype_list = [
            ("torch.FloatTensor", True),
            ("torch.HalfTensor", True),
            ("torch.IntTensor", True),
            ("torch.LongTensor", True),
        ]
        if TEST_BFLOAT16:
            dtype_list.append(("torch.BFloat16Tensor", True))
        list_list = [dtype_list, group]

        for dtype_tuple, src in product(*list_list):
            if not dtype_tuple[1]:
                continue
            ttype = dtype_tuple[0]
            if rank == src:
                tensor = self.to_device(
                    _build_tensor(src + 1, master_value).type(ttype)
                )
                dist.all_reduce(tensor, op)
                # print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess(
                    (tensor.float() - _build_tensor(src + 1, expected_value).to("mlu"))
                    .abs()
                    .cpu()
                    .max()
                    .item(),
                    3e-3,
                )
            else:
                tensor = self.to_device(
                    _build_tensor(src + 1, worker_value).type(ttype)
                )
                dist.all_reduce(tensor, op)
                # print("sum", rank, src, tensor.cpu().view(-1)[0].item())
                self.assertLess(
                    (tensor.float() - _build_tensor(src + 1, expected_value).to("mlu"))
                    .abs()
                    .cpu()
                    .max()
                    .item(),
                    3e-3,
                )

    # @unittest.skip("not test")
    def test_all_reduce_sum(self):
        torch.manual_seed(1)
        group, rank = self._init_global_test()
        a = 10
        b = 2
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.SUM,
            a,
            b,
            a + b * (len(group) - 1),
        )

    # @unittest.skip("not test")
    def test_all_reduce_product(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.PRODUCT,
            10,
            2,
            reduce((lambda x, y: x * y), [2] * (len(group) - 1), 10),
        )

    # @unittest.skip("not test")
    def test_all_reduce_min(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MIN,
            1,
            1010,
            1,
        )

    # @unittest.skip("not test")
    def test_all_reduce_max(self):
        group, rank = self._init_global_test()
        self._test_all_reduce_helper(
            group,
            rank,
            dist.ReduceOp.MAX,
            10,
            -1,
            10,
        )

    # TODO(PYTORCH-12018)
    @unittest.skip("not test")
    def test_all_reduce_max_with_profiler(self):
        group, rank = self._init_global_test()
        with torch.autograd.profiler.profile(use_mlu=True, use_kineto=True) as p:
            self._test_all_reduce_helper(
                group,
                rank,
                dist.ReduceOp.MAX,
                10,
                -1,
                10,
            )

        found_atomic_op = False
        for evt in p.function_events:
            if "Atomic Operation" in evt.name:
                found_atomic_op = True
        if torch.mlu.get_device_properties(torch.mlu.current_device()).major == 3:
            self.assertTrue(found_atomic_op)
        else:
            self.assertFalse(found_atomic_op)

    # @unittest.skip("not test")
    def test_empty_tensors(self):
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        pg = _get_default_group()

        ys = [self.to_device(torch.FloatTensor([]))]
        xs = [[self.to_device(torch.FloatTensor([])) for _ in range(self.world_size)]]
        pg.reduce_scatter(ys, xs).wait()
        self.assertEqual(0, ys[0].numel())

    def _test_scatter_helper(self, group, rank):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        world_size = len(group)
        dst_list = [i for i in range(world_size)]
        async_op_list = [True, False]
        list_list = [dst_list, dtype_list, async_op_list]
        for dst, dtype, async_op in product(*list_list):
            in_tensors = (
                [_build_tensor(dst + 1, i).type(dtype).mlu() for i in group]
                if rank == dst
                else []
            )
            out_tensor = _build_tensor(dst + 1, -1).type(dtype).mlu()
            expected_tensor = _build_tensor(dst + 1, rank).type(dtype)

            work = dist.scatter(out_tensor, in_tensors, src=dst, async_op=async_op)
            if async_op:
                work.wait()
            self.assertEqual(expected_tensor, out_tensor.cpu())

        self._barrier()

    def test_scatter(self):
        group, rank = self._init_global_test()
        self._test_scatter_helper(group, rank)

    def _test_reduce_scatter_helper(self, rank, op, expected_value):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        type_info = [
            ("torch.FloatTensor", True),
            ("torch.HalfTensor", True),
            ("torch.CharTensor", True),
            ("torch.ByteTensor", True),
            ("torch.IntTensor", True),
            ("torch.LongTensor", True),
        ]
        if TEST_BFLOAT16:
            type_info.append(("torch.BFloat16Tensor", True))
        for ttype, is_test in type_info:
            if not is_test:
                continue
            else:
                output = self.to_device(torch.tensor([0]).type(ttype))
                if op == dist.ReduceOp.PRODUCT:
                    tensor_list = [
                        self.to_device(
                            torch.tensor([(rank + i) % self.world_size + 1]).type(ttype)
                        )
                        for i in range(0, self.world_size)
                    ]
                else:
                    tensor_list = [
                        self.to_device(torch.tensor([rank + i]).type(ttype))
                        for i in range(0, self.world_size)
                    ]
                dist.reduce_scatter(output, tensor_list, op)
                # mlu bfloat16 is not support item() now.
                self.assertEqual(
                    expected_value.cpu()
                    if isinstance(expected_value, torch.Tensor)
                    else expected_value,
                    output.cpu() if isinstance(output, torch.Tensor) else output,
                )

    def test_reduce_scatter_tensor_continuous(self):
        _, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            input_tensors = list(
                torch.arange(self.world_size).type(dtype).mlu().chunk(self.world_size)
            )
            output_tensor = torch.zeros(1).type(dtype).mlu()
            expected_tensor = torch.tensor([rank * self.world_size]).type(dtype)
            dist.reduce_scatter(output_tensor, input_tensors, dist.ReduceOp.SUM)
            self.assertEqual(output_tensor.cpu(), expected_tensor)

    def test_reduce_scatter_tensor_not_contiguous(self):
        group, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            input_tensors = [
                torch.tensor([i for i in range(128)])
                .type(dtype)
                .mlu()
                .as_strided([64, 2], [1, 64])
                for i in range(self.world_size)
            ]
            output_tensor = (
                torch.reshape(torch.tensor([0 for i in range(128)]), (64, 2))
                .type(dtype)
                .mlu()
            )
            expected_tensor = (
                (torch.tensor([i * self.world_size for i in range(128)]))
                .type(dtype)
                .as_strided([64, 2], [1, 64])
            )
            dist.reduce_scatter(output_tensor, input_tensors)
            self.assertEqual(output_tensor, expected_tensor)

    def test_reduce_scatter_tensor_not_continuous(self):
        _, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            input_tensors = []
            for i in range(self.world_size):
                input_tensors.append(torch.tensor([i]).type(dtype).mlu())
            output_tensor = torch.zeros(1).type(dtype).mlu()
            expected_tensor = torch.tensor([rank * self.world_size]).type(dtype)
            dist.reduce_scatter(output_tensor, input_tensors, dist.ReduceOp.SUM)
            self.assertEqual(output_tensor.cpu(), expected_tensor)

    # @unittest.skip("not test")
    def test_reduce_scatter_sum(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank,
            dist.ReduceOp.SUM,
            float(self.world_size * (self.world_size - 1) / 2) + rank * self.world_size,
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_min(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(rank, dist.ReduceOp.MIN, float(rank))

    # @unittest.skip("not test")
    def test_reduce_scatter_max(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank, dist.ReduceOp.MAX, float(rank + self.world_size - 1)
        )

    # @unittest.skip("not test")
    def test_reduce_scatter_product(self):
        _, rank = self._init_global_test()
        self._test_reduce_scatter_helper(
            rank, dist.ReduceOp.PRODUCT, float(math.factorial(self.world_size))
        )

    # @unittest.skip("not test")
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE`"
    )
    @largeTensorTest("41GB", device="mlu")
    def test_reduce_scatter_min_large(self):
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        op = dist.ReduceOp.MIN
        expected_value = float(rank)
        size = 1073741824
        output = self.to_device(torch.empty(size, dtype=torch.float))
        tensor_list = [
            self.to_device(torch.empty(size, dtype=torch.float).fill_(rank + i))
            for i in range(0, self.world_size)
        ]
        expected_tensor = torch.empty(size, dtype=torch.float).fill_(expected_value)
        dist.reduce_scatter(output, tensor_list, op)
        self.assertEqual(expected_tensor, output.cpu())

    # @unittest.skip("not test")
    def test_reduce_scatter_tensor(self):
        _, rank = self._init_global_test()
        size = 2
        tensor_out = torch.zeros(size, dtype=torch.int64).mlu(
            rank % torch.mlu.device_count()
        )

        # Concatenated input
        tensor_in = torch.arange(self.world_size * size).mlu(
            rank % torch.mlu.device_count()
        )
        dist.reduce_scatter_tensor(tensor_out, tensor_in)
        # Check result
        expected_tensor = torch.arange(rank * size, (rank + 1) * size) * self.world_size
        self.assertEqual(tensor_out.cpu(), expected_tensor)

        # Stacked input
        tensor_out = torch.zeros(size, dtype=torch.int64).mlu(
            rank % torch.mlu.device_count()
        )
        tensor_in = torch.reshape(tensor_in, (self.world_size, size)).mlu(
            rank % torch.mlu.device_count()
        )
        dist.reduce_scatter_tensor(tensor_out, tensor_in)
        # Check result
        # Should be the same as the result in concatenated case
        self.assertEqual(tensor_out.cpu(), expected_tensor)

    def _test_all_gather_helper(self, group, rank, times=1):
        def _build_tensor(size, value):
            return (
                torch.arange(value, size * size * size + value)
                .view(size, size, size)
                .float()
            )

        torch.mlu.set_device(rank % torch.mlu.device_count())
        loop_list = list(range(times))
        dtype_list = [
            ("torch.FloatTensor", True),
            ("torch.HalfTensor", True),
            ("torch.CharTensor", True),
            ("torch.ByteTensor", True),
            ("torch.IntTensor", True),
            ("torch.LongTensor", True),
            ("torch.DoubleTensor", True),
            ("torch.BoolTensor", True),
        ]
        if TEST_BFLOAT16:
            dtype_list.append(("torch.BFloat16Tensor", True))
        list_list = [loop_list, dtype_list, group]
        for _, dtype_tuple, src in product(*list_list):
            if not dtype_tuple[1]:
                continue
            ttype = dtype_tuple[0]
            tensor = self.to_device(_build_tensor(src + 1, rank).type(ttype))
            tensors = [
                self.to_device(_build_tensor(src + 1, -1).type(ttype)) for i in group
            ]
            dist.all_gather(tensors, tensor)
            expected_tensors = [
                self.to_device(_build_tensor(src + 1, i).type(ttype)) for i in group
            ]
            for t1, t2 in zip(tensors, expected_tensors):
                self.assertTrue(t1.cpu().eq(t2.cpu()).min().item())

    # @unittest.skip("not test")
    def test_all_gather(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank)

    def test_all_gather_tensor_continuous(self):
        group, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            output_tensors = list(
                torch.zeros(self.world_size).type(dtype).mlu().chunk(self.world_size)
            )
            input_tensor = torch.tensor([rank]).type(dtype).mlu()
            expected_tensors = list(
                torch.arange(self.world_size).type(dtype).chunk(self.world_size)
            )
            dist.all_gather(output_tensors, input_tensor)
            for t1, t2 in zip(output_tensors, expected_tensors):
                self.assertEqual(t1.cpu(), t2)

    # This test set tenors in output not contiguous itself
    # @unittest.skip("not test")
    def test_all_gather_tensor_not_contiguous(self):
        group, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            output_tensors = [
                torch.zeros(128).type(dtype).mlu().as_strided([64, 2], [1, 64])
                for i in range(self.world_size)
            ]
            input_tensor = torch.tensor([i for i in range(128)]).type(dtype).mlu()
            expected_tensors = [
                torch.reshape(
                    (torch.tensor([i for i in range(128)])).type(dtype), (64, 2)
                )
                for i in range(self.world_size)
            ]
            dist.all_gather(output_tensors, input_tensor)
            for t1, t2 in zip(output_tensors, expected_tensors):
                self.assertEqual(t1.cpu(), t2)

    # This test set output_tensors not continuous to next one
    # @unittest.skip("not test")
    def test_all_gather_tensor_not_continuous(self):
        group, rank = self._init_global_test()
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        torch.mlu.set_device(rank % torch.mlu.device_count())
        for dtype in dtype_list:
            output_tensors = []
            for i in range(self.world_size):
                output_tensors.append(torch.zeros(1).type(dtype).mlu())
            input_tensor = torch.tensor([rank]).type(dtype).mlu()
            expected_tensors = list(
                torch.arange(self.world_size).type(dtype).chunk(self.world_size)
            )
            dist.all_gather(output_tensors, input_tensor)
            for t1, t2 in zip(output_tensors, expected_tensors):
                self.assertEqual(t1.cpu(), t2)

    # @unittest.skip("not test")
    def test_all_gather_object(self):
        _, rank = self._init_global_test()
        next_rank = (rank + 1) % torch.mlu.device_count()
        torch.mlu.set_device(next_rank)

        f = Foo(10)
        f.bar = 1  # pylint: disable=C0104,W0201
        torch.manual_seed(0)
        foo_cpu_tensor = Foo(torch.randn(3, 3))
        foo_mlu_tensor = Foo(torch.randn(3, 3).mlu(0))

        gather_objects = [
            {"key1": 3, "key2": 4, "key3": {"nested": True}},
            f,
            foo_cpu_tensor,
            foo_mlu_tensor,
            "foo",
            [1, 2, True, "string", [4, 5, "nested"]],
        ]

        output_gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(
            output_gathered, gather_objects[rank % len(gather_objects)]
        )

        for i, val in enumerate(output_gathered):
            expected = gather_objects[i % len(gather_objects)]
            self.assertEqual(val, expected)

            output_gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(
                output_gathered, gather_objects[rank % len(gather_objects)]
            )

    # @unittest.skip("not test")
    def test_all_gather_into_tensor_ops(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = torch.tensor([self.rank]).mlu()
        output_t = torch.zeros((self.world_size), dtype=tensor.dtype).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

    # @unittest.skip("not test")
    def test_all_gather_into_cat_tensor(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        size = 2
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = (torch.ones([size, size]) * self.rank).mlu()
        output_t = (
            torch.ones([self.world_size * size, size], dtype=tensor.dtype) * (-1)
        ).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Check result
        # Concatenate all blocks into a bigger tensor
        expected_tensor = torch.cat(
            [torch.ones([size, size]) * i for i in range(self.world_size)]
        )
        # Verification
        self.assertEqual(output_t, expected_tensor)

    # @unittest.skip("not test")
    def test_all_gather_into_stack_tensor(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        size = 2
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = (torch.ones([size, size]) * self.rank).mlu()
        output_t = (
            torch.ones([self.world_size, size, size], dtype=tensor.dtype) * (-1)
        ).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Check result
        # Concatenate all blocks into a bigger tensor
        expected_tensor = torch.stack(
            [torch.ones([size, size]) * i for i in range(self.world_size)]
        )
        # Verification
        self.assertEqual(output_t, expected_tensor)

    # @unittest.skip("not test")
    def test_all_gather_into_tensor_basics(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        # anticpate an error
        with self.assertRaisesRegex(
            RuntimeError,
            "output tensor size must be equal to \
world_size times input tensor size",
        ):
            tensor = torch.tensor([self.rank]).mlu()
            output_t = torch.zeros((self.world_size + 1), dtype=tensor.dtype).mlu()
            # fails the check because output_t is not correctly sized
            dist.all_gather_into_tensor(output_t, tensor)

        # anticpate an error
        with self.assertRaisesRegex(
            RuntimeError,
            "output tensor must have the same type \
as input tensor",
        ):
            tensor = torch.tensor([self.rank], dtype=torch.float).mlu()
            output_t = torch.zeros((self.world_size + 1), dtype=torch.long).mlu()
            # fails the check because the dtype is different
            dist.all_gather_into_tensor(output_t, tensor)

    def test_reduce_scatter_v(self):
        self._barrier()
        group, rank = self._init_global_test()
        device_id = rank % torch.mlu.device_count()

        input_split_sizes = []
        for src in group:
            input_split_sizes.append(src + 1)
        start_len = sum(input_split_sizes[:rank])
        end_len = start_len + input_split_sizes[rank]
        sum_len = sum(input_split_sizes)
        master_value = 2
        worker_value = 10

        for async_val in [True, False]:
            tensor = _build_tensor(sum_len, worker_value, device_id=device_id)
            tensor[start_len:end_len].fill_(master_value)
            out_tensor = (
                torch.empty(
                    input_split_sizes[rank], sum_len, sum_len, dtype=torch.float
                )
                .fill_(-1)
                .mlu(device_id)
            )

            req = dist.reduce_scatter(
                out_tensor,
                list(torch.split(tensor, input_split_sizes)),
                dist.ReduceOp.SUM,
                async_op=async_val,
            )
            if async_val:
                req.wait()

            expected_value = 2 + (10 * (len(group) - 1))
            expected_tensor = torch.empty(
                input_split_sizes[rank], sum_len, sum_len, dtype=torch.float
            )
            expected_tensor = expected_tensor.fill_(expected_value).mlu(device_id)

            self.assertEqual(out_tensor, expected_tensor)
        self._barrier()

    def test_all_gather_v(self):
        self._barrier()
        group, rank = self._init_global_test()
        device_id = rank % torch.mlu.device_count()

        output_split_sizes = []
        for dst in group:
            output_split_sizes.append(dst + 1)
        sum_len = sum(output_split_sizes)
        value = 2

        for async_val in [True, False]:
            tensor = (
                torch.empty(
                    output_split_sizes[rank], sum_len, sum_len, dtype=torch.float
                )
                .fill_(value)
                .mlu(device_id)
            )
            out_tensor = _build_tensor(sum_len, -1, device_id=device_id)

            req = dist.all_gather(
                list(torch.split(out_tensor, output_split_sizes)),
                tensor,
                async_op=async_val,
            )
            if async_val:
                req.wait()

            expected_value = value
            expected_tensor = _build_tensor(
                sum_len, expected_value, device_id=device_id
            )

            self.assertEqual(out_tensor, expected_tensor)
        self._barrier()

    # @unittest.skip("not test")
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE`"
    )
    @largeTensorTest("45GB", device="mlu")
    def test_all_gather_into_cat_tensor_large(self):
        device_id = self.rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        size = 1073741824
        # all_gather_into_tensor is device number agnostic.
        # Each rank contribute one tensor regardless of MLU counts
        tensor = (torch.ones([size, 2]) * self.rank).mlu()
        output_t = (
            torch.ones([self.world_size * size, 2], dtype=tensor.dtype) * (-1)
        ).mlu()

        dist.all_gather_into_tensor(output_t, tensor)

        # Check result
        # Concatenate all blocks into a bigger tensor
        expected_tensor = torch.cat(
            [torch.ones([size, 2]) * i for i in range(self.world_size)]
        )
        # Verification
        self.assertEqual(output_t, expected_tensor)

    # @unittest.skip("not test")
    def test_pressure(self):
        group, rank = self._init_global_test()
        self._test_all_gather_helper(group, rank, times=20)

    def _test_gather_helper(self, group, rank):
        torch.mlu.set_device(rank % torch.mlu.device_count())
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        world_size = len(group)
        dst_list = [i for i in range(world_size)]
        async_op_list = [True, False]
        list_list = [dst_list, dtype_list, async_op_list]
        for dst, dtype, async_op in product(*list_list):
            in_tensor = _build_tensor(dst + 1, rank).type(dtype).mlu()
            out_tensors = (
                [_build_tensor(dst + 1, -1).type(dtype).mlu() for i in group]
                if rank == dst
                else []
            )
            work = None
            if rank == dst:
                work = dist.gather(in_tensor, out_tensors, dst=dst, async_op=async_op)
            else:
                dist.gather(in_tensor, dst=dst, async_op=async_op)
            if rank == dst:
                if async_op:
                    work.wait()
                expected_tensors = [
                    _build_tensor(dst + 1, i).type(dtype).mlu() for i in group
                ]
                for out, expt in zip(out_tensors, expected_tensors):
                    self.assertEqual(out.cpu(), expt)

    # @unittest.skip("not test")
    def test_gather(self):
        group, rank = self._init_global_test()
        self._test_gather_helper(group, rank)

    # @unittest.skip("not test")
    def test_gather_object(self):
        _, rank = self._init_global_test()
        next_rank = (rank + 1) % torch.mlu.device_count()
        torch.mlu.set_device(next_rank)

        f = Foo(10)
        f.bar = 1  # pylint: disable=C0104,W0201
        torch.manual_seed(0)
        foo_cpu_tensor = Foo(torch.randn(3, 3))
        foo_mlu_tensor = Foo(torch.randn(3, 3).mlu(0))

        gather_objects = [
            {"key1": 3, "key2": 4, "key3": {"nested": True}},
            f,
            foo_cpu_tensor,
            foo_mlu_tensor,
            "foo",
            [1, 2, True, "string", [4, 5, "nested"]],
        ]

        world_size = dist.get_world_size()
        output_gathered = [None for _ in range(world_size)]
        for dst in range(world_size):
            dist.gather_object(
                gather_objects[dist.get_rank()],
                output_gathered if dist.get_rank() == dst else None,
                dst=dst,
            )

            if rank == dst:
                for i, val in enumerate(output_gathered):
                    expected = gather_objects[i % len(gather_objects)]
                    self.assertEqual(val, expected)

    def _test_p2pop_helper(self, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        dist.barrier()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        for ttype in dtype_list:
            send_tensor = torch.tensor(range(10)).type(ttype).to("mlu")
            recv_tensor = torch.zeros(10).type(ttype).to("mlu")
            p2p_op_list = []
            if rank == 0:
                p2p_op_list.append(dist.P2POp(dist.isend, send_tensor, 1))
            elif rank == 1:
                p2p_op_list.append(dist.P2POp(dist.irecv, recv_tensor, 0))
            if rank in [0, 1]:
                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()
            dist.barrier()
            if rank == 1:
                self.assertEqual(recv_tensor.float().cpu(), send_tensor.float().cpu())

    # @unittest.skip("not test")
    def test_p2pop(self):
        _, rank = self._init_global_test()
        self._test_p2pop_helper(rank)

    # @unittest.skip("not test")
    def test_batch_isend_irecv(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        world_size = dist.get_world_size()
        recv_tensors = [None for _ in range(world_size)]
        expected_tensors = [None for _ in range(world_size)]

        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        for val in ["1", "0"]:
            p2p_op_list = []
            os.environ["CNCL_BLOCKING_WAIT"] = val
            for src in range(0, dist.get_world_size()):
                send_tensor = _build_tensor(rank + 1, device_id=device_id).fill_(src)
                recv_tensors[src] = _build_tensor(
                    src + 1, value=-1, device_id=device_id
                )
                expected_tensors[src] = _build_tensor(
                    src + 1, value=-1, device_id=device_id
                ).fill_(rank)
                recv_op = dist.P2POp(dist.irecv, recv_tensors[src], src)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, src)
                p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

            for src in range(0, world_size):
                self.assertEqual(recv_tensors[src], expected_tensors[src])

        dist.barrier()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_cncl_self(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        p2p_op_list = []

        if rank == 0:
            send_tensor = _build_tensor(rank + 1, device_id=device_id)
            recv_tensor = _build_tensor(rank + 1, value=-1, device_id=device_id)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, 0)
            p2p_op_list.append(recv_op)
            send_op = dist.P2POp(dist.isend, send_tensor, 0)
            p2p_op_list.append(send_op)

            reqs = dist.batch_isend_irecv(p2p_op_list)
            for req in reqs:
                req.wait()

        dist.barrier()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_tensor_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        if rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "No backend type associated with device type cpu"
            ):
                send_tensor = _build_tensor(rank + 1)
                send_op = dist.P2POp(dist.isend, send_tensor, 1)
                req = dist.batch_isend_irecv([send_op])
                req.wait()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_op_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        if rank == 0:
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            with self.assertRaisesRegex(RuntimeError, "^Invalid ``op``"):
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                send_op = dist.P2POp(dist.broadcast, send_tensor, 1)
                req = dist.batch_isend_irecv([send_op])
                req.wait()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_op_list_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        if rank == 0:
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            with self.assertRaisesRegex(RuntimeError, "^Invalid ``p2p_op_list``"):
                send_tensor = _build_tensor(rank + 1)
                req = dist.batch_isend_irecv([1, 2])
                req.wait()

        dist.barrier()

    # @unittest.skip("not test")
    def test_batch_isend_irecv_mixed_backend_err(self):
        _, rank = self._init_global_test()
        dist.barrier()
        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)
        group_gloo = dist.new_group(ranks=[0, 1], backend="gloo")
        group_cncl = dist.new_group(ranks=[0, 1], backend="cncl")
        if rank == 0:
            with self.assertRaisesRegex(
                RuntimeError, "All ops need to use the same group"
            ):
                send_tensor = _build_tensor(rank + 1)
                send_op_gloo = dist.P2POp(dist.isend, send_tensor, 1, group_gloo)
                send_op_cncl = dist.P2POp(dist.isend, send_tensor, 1, group_cncl)
                req = dist.batch_isend_irecv([send_op_gloo, send_op_cncl])
                req.wait()

    def test_batch_isend_irecv_ring_exchange_cncl(self):
        _, rank = self._init_global_test()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        device_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(device_id)

        send_tensor = _build_tensor(world_size, device_id=device_id)
        recv_tensor = _build_tensor(world_size, value=-1, device_id=device_id)
        send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
        recv_op = dist.P2POp(
            dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
        )
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        for req in reqs:
            req.wait()

        dist.barrier()

    def test_batch_isend_irecv_no_rank_zero_cncl(self):
        _, rank = self._init_global_test()
        world_size = dist.get_world_size()
        # Ensure the process group has been fully initialized (needed by
        # the first sub-group batch_isend_irecv call)
        dist.barrier()
        if world_size > 2:
            rank = dist.get_rank()
            device_id = rank % torch.mlu.device_count()
            torch.mlu.set_device(device_id)
            p2p_op_list = []

            if rank == 1:
                peer = 2
            elif rank == 2:
                peer = 1

            if rank in [1, 2]:
                send_tensor = _build_tensor(rank + 1, device_id=device_id)
                recv_tensor = _build_tensor(peer + 1, value=-1, device_id=device_id)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, peer)
                p2p_op_list.append(recv_op)
                send_op = dist.P2POp(dist.isend, send_tensor, peer)
                p2p_op_list.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_op_list)
                for req in reqs:
                    req.wait()

            dist.barrier()

    def _test_barrier_helper(self, group, rank):
        dist.barrier()  # test barrier before set device
        torch.mlu.set_device(rank % torch.mlu.device_count())
        WAIT_TIME = 10  # seconds

        # Because MLU does not support Double currently, the precision of the float cast result
        # of time.time() is not enough, so we remainder the value by 100000
        for src in group:
            expected_time = self.to_device(torch.FloatTensor(1).fill_(0.0))
            if src == rank:
                expected_time.fill_(time.time() % 100000 + WAIT_TIME)
                dist.broadcast(expected_time, src)
                time.sleep(WAIT_TIME + 0.1)
                dist.barrier()
            else:
                dist.broadcast(expected_time, src)
                dist.barrier()
                finish_time = time.time() % 100000
                self.assertGreaterEqual(
                    float(finish_time),
                    float(expected_time.item()),
                    "destination rank: %d, my rank: %d" % (src, rank),
                )

    # @unittest.skip("not test")
    def test_barrier(self):
        group, rank = self._init_global_test()
        self._test_barrier_helper(group, rank)

    def _test_all_to_all_single_equal_split_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)

        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        for ttype in dtype_list:
            in_tensor = (torch.ones([size, size]) * rank).type(ttype).mlu()
            expected_tensor = torch.cat(
                [torch.ones([1, size]) * i for i in group]
            ).type(ttype)
            out_tensor = (torch.ones([size, size]) * -1).type(ttype).mlu()
            dist.all_to_all_single(out_tensor, in_tensor)
            self.assertEqual(out_tensor.cpu(), expected_tensor)

    def _test_all_to_all_single_unequal_split_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        in_splits = [i + 1 for i in group]
        out_splits = [rank + 1 for _ in group]
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        for ttype in dtype_list:
            in_tensor = (torch.ones([sum(in_splits), size]) * rank).type(ttype).mlu()
            out_tensor = (torch.zeros([(rank + 1) * size, size])).type(ttype).mlu()
            expected_tensor = torch.cat(
                [torch.ones([rank + 1, size]) * i for i in group]
            ).type(ttype)
            dist.all_to_all_single(out_tensor, in_tensor, out_splits, in_splits)
            self.assertEqual(out_tensor.cpu(), expected_tensor)

    def _test_all_to_all_helper(self, group, rank):
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        in_splits = [i + 1 for i in group]
        dtype_list = [
            "torch.FloatTensor",
            "torch.HalfTensor",
            "torch.CharTensor",
            "torch.ByteTensor",
            "torch.IntTensor",
            "torch.LongTensor",
            "torch.DoubleTensor",
            "torch.BoolTensor",
        ]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        for ttype in dtype_list:
            in_tensors = [
                (torch.ones([in_splits[i], size]) * rank).type(ttype).mlu()
                for i in group
            ]
            out_tensors = [
                torch.zeros([(rank + 1), size]).type(ttype).mlu() for _ in group
            ]
            expected_tensors = [
                (torch.ones([rank + 1, size]) * i).type(ttype) for i in group
            ]
            dist.all_to_all(out_tensors, in_tensors)
            for out, expt in zip(out_tensors, expected_tensors):
                self.assertEqual(out.cpu(), expt)

    def test_all_to_all_single_equal_large(self):
        group, rank = self._init_global_test()
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = len(group)
        num = int(1073741824 / size)
        dtype_list = ["torch.FloatTensor"]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        for ttype in dtype_list:
            in_tensor = (torch.ones([size, num]) * rank).type(ttype).mlu()
            out_tensor = (torch.ones([size, num]) * -1).type(ttype).mlu()
            dist.all_to_all_single(out_tensor, in_tensor)
        torch.mlu.synchronize()

    def test_all_to_all_single_unequal_large(self):
        group, rank = self._init_global_test()
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())

        in_splits = [i + 1 for i in group]
        size = int(1073741824 / sum(in_splits))
        out_splits = [rank + 1 for _ in group]
        dtype_list = ["torch.FloatTensor"]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")

        for ttype in dtype_list:
            in_tensor = (torch.ones([sum(in_splits), size]) * rank).type(ttype).mlu()
            out_tensor = (
                (torch.zeros([(rank + 1) * len(group), size])).type(ttype).mlu()
            )
            dist.all_to_all_single(out_tensor, in_tensor, out_splits, in_splits)
        torch.mlu.synchronize()

    def test_all_to_all_large(self):
        group, rank = self._init_global_test()
        os.environ["CNCL_MEM_POOL_MULTI_CLIQUE_ENABLE"] = "1"
        torch.mlu.set_device(rank % torch.mlu.device_count())
        size = 1073741824
        in_splits = [i + 1 for i in group]
        dtype_list = ["torch.FloatTensor"]
        if TEST_BFLOAT16:
            dtype_list.append("torch.BFloat16Tensor")
        for ttype in dtype_list:
            in_tensors = list(
                torch.chunk(
                    (torch.ones(int(size)) * rank).type(ttype).mlu(), len(group)
                )
            )
            out_tensors = list(
                torch.chunk(torch.zeros(int(size)).type(ttype).mlu(), len(group))
            )
            dist.all_to_all(out_tensors, in_tensors)
        torch.mlu.synchronize()

    # @unittest.skip("not test")
    def test_all_to_all_single_equal_split(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_single_equal_split_helper(group, rank)

    # @unittest.skip("not test")
    def test_all_to_all_single_unequal_split(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_single_unequal_split_helper(group, rank)

    # @unittest.skip("not test")
    def test_all_to_all(self):
        group, rank = self._init_global_test()
        self._test_all_to_all_helper(group, rank)

    def test_ddp_grad_div_uneven_inputs(self):
        # Test gradient division during training with join() API. If
        # divide_by_initial_world_size=False, we scale by the effective world
        # size when allreducing grads.
        dim = 5
        batch = 1
        grad_scale = 50
        model = nn.Linear(dim, dim, bias=False)
        inp = torch.ones(batch, dim, device=self.rank) * grad_scale
        net = torch.nn.parallel.DistributedDataParallel(
            model.mlu(self.rank), device_ids=[self.rank], bucket_cap_mb=1
        )
        n_iters = 3
        if self.rank > 0:
            n_iters += 2

        with net.join(divide_by_initial_world_size=False):
            for _ in range(n_iters):
                loss = net(inp).sum()
                loss.backward()
                # The grad is always expected_grad, since we divide by the number
                # of currently active processes and inactive processes contribute
                # zero gradient. If we kept dividing by static initial world
                # size as processes leave, the grad would be smaller.
                expected_grad = torch.ones(dim, dim, device=self.rank) * grad_scale
                param = list(net.parameters())[0]
                self.assertEqual(expected_grad, param.grad)
                # Avoid accumulating grads so that it's the same every iteration
                net.zero_grad()
                torch.mlu.synchronize(device=self.rank)

        # If divide_by_initial_world_size=True (default), we always scale grads
        # by the initial world_size.
        with net.join(divide_by_initial_world_size=True):
            for i in range(n_iters):
                loss = net(inp).sum()
                loss.backward()
                effective_ws = dist.get_world_size()
                if i >= 3:
                    effective_ws -= 1
                expected_grad = (
                    torch.ones(dim, dim, device=self.rank) * grad_scale * effective_ws
                ) / dist.get_world_size()
                param = list(net.parameters())[0]
                self.assertEqual(expected_grad, param.grad)
                # Avoid accumulating grad so that it's the same every iteration.
                net.zero_grad()
                torch.mlu.synchronize(device=self.rank)

    @classmethod
    def _model_step(cls, model):
        for param in model.parameters():
            if param.grad is not None:
                param.data = param.data + param.grad
                param.grad.detach_()
                param.grad.zero_()

    # END TO END TEST FOR DISTRIBUTEDDATAPARALLEL
    @classmethod
    def _test_DDP_helper(cls, model, input_var, target, loss, flg, scale_factor=1.0):
        model.train()
        output = model(input_var, flg)
        l = loss(output, target) * scale_factor
        l.backward()

    def _assert_equal_param(self, param, param_DDP):
        self.assertEqual(len(param), len(param_DDP))
        for p, p_DDP in zip(param, param_DDP):
            self.assertEqual(p, p_DDP.cpu())

    def _test_multi_nodes_helper(self, param_DDP, rank):
        ps = []
        file_name = "params_" + str(self.world_size) + "cards.pt"
        single_node_params_file = os.path.join(TEMP_DIR, file_name)
        if self.nnodes == 1:
            if rank == 0:
                for p in param_DDP:
                    ps.append(p.cpu())
                torch.save(ps, single_node_params_file)
        else:
            if os.path.exists(single_node_params_file):
                ps = torch.load(
                    single_node_params_file, map_location=torch.device("cpu")
                )
                for p_sing, p_mult in zip(ps, param_DDP):
                    self.assertEqual(p_sing, p_mult.cpu())
            else:
                print(
                    "WARNING: "
                    + single_node_params_file
                    + " not found, if you want to "
                    "compare with single mlu card parameters of Net, please run single "
                    "node version of test_distributed.py firstly!"
                )

    def _test_DDP_5iter(
        self,
        model_base,
        model_DDP,
        input_data,
        target,
        loss,
        local_bs,
        rank,
        batch_size,
        base_is_mlu=False,
        offset=None,
    ):
        for _ in range(5):
            # single cpu training
            self._test_DDP_helper(model_base, input_data, target, loss, base_is_mlu)

            # DDP training, DDP scatters subsets of input_cpu to nodes/MLUs
            if offset is None:
                offset = rank * local_bs
            self._test_DDP_helper(
                model_DDP,
                input_data[offset : offset + local_bs],
                target[offset : offset + local_bs],
                loss,
                True,
                dist.get_world_size() * local_bs / batch_size,
            )

            # Update weights and run a second iteration to shake out errors
            self._model_step(model_base)
            self._model_step(model_DDP)
            self._assert_equal_param(
                list(model_base.parameters()), list(model_DDP.module.parameters())
            )

            # Shuffle the input so that DDP input is different
            input_data = input_data[torch.randperm(batch_size)]
        self._test_multi_nodes_helper(list(model_DDP.module.parameters()), rank)

    def _test_DistributedDataParallel(self, rank):
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())

        # cpu training setup
        model = Net()
        # model.fc1.weight.register_hook(hook)

        # DDP training setup
        model_DDP = copy.deepcopy(model)
        model_DDP.to("mlu")
        # can use find_unused_parameters=True
        model_DDP = nn.parallel.DistributedDataParallel(
            model_DDP, device_ids=[rank % torch.mlu.device_count()]
        )

        def hook(grad):  # pylint: disable=W0612
            print(
                "hook no_grad_param: ",
                model_DDP.module.no_grad_param.size(),
                model_DDP.module.no_grad_param.cpu(),
            )
            return grad

        # model_DDP.module.fc1.weight.register_hook(hook)

        # dummy data initialization
        local_bs = 1
        global_bs = self.nproc_per_node * self.nnodes * local_bs
        input_cpu = torch.randn(global_bs, 2)
        target = torch.randn(global_bs, 4)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model,
            model_DDP,
            input_cpu,
            target,
            loss,
            local_bs,
            rank,
            global_bs,
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)

    # @unittest.skip("not test")
    def test_distributedDataParallel_side_stream(self):
        torch.manual_seed(1)
        os.environ["PYTORCH_DDP_USE_SIDE_STREAM"] = "0"
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)
        os.environ["PYTORCH_DDP_USE_SIDE_STREAM"] = "1"
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)
        try:
            os.environ.pop("PYTORCH_DDP_USE_SIDE_STREAM")
        except Exception as e:
            # Ignore Errors of os.environ.
            pass

    # @unittest.skip("not test")
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_distributedDataParallel_logger(self):
        torch.manual_seed(1)
        out = OutputGrabber(sys.stderr)
        out.start()
        _, rank = self._init_global_test()
        self._test_DistributedDataParallel(rank)
        out.stop()
        self.assertTrue(
            "Warning: Time stats are currently only collected"
            " for CPU and CUDA devices." not in out.capturedtext
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel_non_default_stream(self):
        _, rank = self._init_global_test()
        dev_id = rank % torch.mlu.device_count()
        torch.mlu.set_device(dev_id)
        queue = torch.mlu.Stream()
        with torch.mlu.stream(queue):
            net = torch.nn.parallel.DistributedDataParallel(
                torch.nn.Linear(1, 1, bias=False).mlu(dev_id), device_ids=[rank]
            )
            for i in range(1000):
                # Clear gradients manually
                grad = net.module.weight.grad
                if grad is not None:
                    grad.detach_()
                    grad.zero_()
                # Forward + BW
                batch = torch.tensor([rank]).float().mlu(dev_id)
                loss = net(batch).sum()
                loss.backward()
                # For each worker, the gradient on the weight should be worker_rank.
                grad = net.module.weight.grad
                avg = grad.clone()
                # All-reducing the gradient averages should give us the gradient
                # average. If not, then one of the workers has not correctly
                # written back the averaged gradient before this all-reduce call.
                dist.all_reduce(avg)
                world_size = self.world_size
                avg.div_(world_size)
                expected_grad = sum(i for i in range(world_size)) / world_size
                self.assertEqual(
                    avg[0, 0],
                    expected_grad,
                    msg=f"Expected gradient of {expected_grad} but got {avg} on rank {rank}",
                )

    def _test_DistributedDataParallel_SyncBatchNorm(
        self, rank, model, size_i, size_t, diff_input_bs=False
    ):
        # mlu training setup
        model_mlu = copy.deepcopy(model)
        model_mlu.to("mlu")

        # DDP training setup
        model_DDP = nn.SyncBatchNorm.convert_sync_batchnorm(copy.deepcopy(model))
        model_DDP.to("mlu")
        model_DDP = nn.parallel.DistributedDataParallel(
            model_DDP, device_ids=[rank % torch.mlu.device_count()]
        )

        # dummy data initialization
        local_bs = rank + 2 if diff_input_bs else 2
        bs_offset = int((rank + 3) * rank / 2) if diff_input_bs else None
        global_bs = (
            int((self.world_size + 3) * self.world_size / 2)
            if diff_input_bs
            else self.world_size * local_bs
        )
        input_cpu = torch.randn(global_bs, *size_i)
        target = torch.randn(global_bs, *size_t)
        loss = nn.MSELoss()

        # check two model parameters over 5 iterations
        self._test_DDP_5iter(
            model_mlu,
            model_DDP,
            input_cpu,
            target,
            loss,
            local_bs,
            rank,
            global_bs,
            True,
            offset=bs_offset,
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet()
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [4])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_No_Affine(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet(False)
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [4])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_2D_Input(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = OnlyBatchNormNet(nn.BatchNorm1d(2))
        self._test_DistributedDataParallel_SyncBatchNorm(rank, model, [2], [2])

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_5D_Input(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = OnlyBatchNormNet(nn.BatchNorm3d(99))
        self._test_DistributedDataParallel_SyncBatchNorm(
            rank, model, [99, 10, 215, 7], [99, 10, 215, 7]
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Gradient(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        # Run a simple end to end DDP model, use result of single node model
        # as baseline
        torch.mlu.set_device(rank % torch.mlu.device_count())
        # training model setup
        model = BatchNormNet()
        self._test_DistributedDataParallel_SyncBatchNorm(
            rank, model, [2], [4], diff_input_bs=True
        )

    # @unittest.skip("not test")
    def test_distributedDataParallel_SyncBatchNorm_Diff_Input_Sizes_Running_Value(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        model = nn.SyncBatchNorm(2, momentum=0.99).to("mlu")
        model_ddp = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        input_var = []
        for i in range(self.world_size):
            input_var_rank = torch.cat(
                [
                    torch.ones(2, 1, 2 ** (i + 1)) * (0.1 ** (i - 1)),
                    torch.ones(2, 1, 2 ** (i + 1)) * (0.3 ** (i - 1)),
                ],
                dim=1,
            )
            input_var.append(input_var_rank)

        all_input_var = torch.cat(
            [
                x.permute(1, 0, 2).contiguous().view(model.num_features, -1)
                for x in input_var
            ],
            dim=1,
        )

        for i in range(100):
            y = model_ddp(input_var[rank].to("mlu"))
            y.mean().backward()

        running_mean, running_var = (
            model_ddp.module.running_mean,
            model_ddp.module.running_var,
        )
        torch.testing.assert_allclose(running_mean.cpu(), all_input_var.cpu().mean(1))
        torch.testing.assert_allclose(running_var.cpu(), all_input_var.cpu().var(1))

    # @unittest.skip("not test")
    def test_distributedDataParallel_cncl_stream(self):
        torch.manual_seed(1)
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())

        model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 5)).to(rank)
        ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

        if rank < 2:
            new_group = dist.new_group([0, 1], backend="cncl")
        else:
            new_group = None

        for epoch in range(3):
            if epoch == 1:
                if rank == 0:
                    cncl_streams_0 = torch.mlu.cncl_stream(f"mlu:{rank}")
                    # print(cncl_streams_0)
                if rank == 1:
                    cncl_streams_1 = torch.mlu.cncl_stream(f"mlu:{rank}")
                    # print(cncl_streams_1)
                if rank == 2:
                    cncl_streams_2 = torch.mlu.cncl_stream(f"mlu:{rank}")
                    # print(cncl_streams_2)
                if rank == 3:
                    cncl_streams_3 = torch.mlu.cncl_stream(f"mlu:{rank}")
                    # print(cncl_streams_3)

            optimizer.zero_grad()
            outputs = ddp_model(torch.randn(20, 10))
            labels = torch.randn(20, 5).to(rank)
            loss_fn(outputs, labels).backward()
            optimizer.step()

            if new_group is not None:
                custom_tensor = torch.tensor(
                    [(rank + 1) * 2], dtype=torch.float, device=f"mlu:{rank}"
                )
                dist.all_reduce(custom_tensor, group=new_group)

        if rank == 0:
            self.assertEqual(len(cncl_streams_0), 2)
        if rank == 1:
            self.assertEqual(len(cncl_streams_1), 2)
        if rank == 2:
            self.assertEqual(len(cncl_streams_2), 1)
        if rank == 3:
            self.assertEqual(len(cncl_streams_3), 1)

    def test_coalescing_manager(self):
        self._barrier()
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        num_colls = 2
        size_per_coll = 8
        small_tensors = [torch.ones(size_per_coll).mlu() for _ in range(num_colls)]

        with dist._coalescing_manager():
            for i in range(num_colls):
                dist.all_reduce(small_tensors[i])

        big_tensor = torch.ones(num_colls * size_per_coll).mlu()
        dist.all_reduce(big_tensor)

        for i in range(num_colls):
            self.assertEqual(
                small_tensors[i],
                big_tensor[i * size_per_coll : (i + 1) * size_per_coll],
            )

        self._barrier()

        # Coalescing manager (async mode)

    def test_coalescing_manager_async(self):
        self._barrier()
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        num_colls = 2
        size_per_coll = 8
        small_tensors = [torch.ones(size_per_coll).mlu() for _ in range(num_colls)]

        with dist._coalescing_manager(async_ops=True) as cm:
            for i in range(num_colls):
                dist.all_reduce(small_tensors[i])
        cm.wait()

        big_tensor = torch.ones(num_colls * size_per_coll).mlu()
        dist.all_reduce(big_tensor)

        for i in range(num_colls):
            self.assertEqual(
                small_tensors[i],
                big_tensor[i * size_per_coll : (i + 1) * size_per_coll],
            )

        self._barrier()

    def test_all_gather_into_tensor_coalesced(self):
        self._barrier()
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        num_allgather = 2
        size_per_allgather = 8
        small_in_tensors = [
            torch.ones(size_per_allgather).mlu() for _ in range(num_allgather)
        ]
        small_out_tensors = [
            torch.zeros(size_per_allgather * self.world_size).mlu()
            for _ in range(num_allgather)
        ]
        with dist._coalescing_manager() as cm:
            for i in range(num_allgather):
                dist.all_gather_into_tensor(small_out_tensors[i], small_in_tensors[i])
        big_in_tensor = torch.ones(num_allgather * size_per_allgather).mlu()
        big_out_tensor = torch.zeros(
            num_allgather * size_per_allgather * self.world_size
        ).mlu()
        dist.all_gather_into_tensor(big_out_tensor, big_in_tensor)
        for i in range(num_allgather):
            self.assertEqual(
                small_out_tensors[i],
                big_out_tensor[
                    i
                    * size_per_allgather
                    * self.world_size : (i + 1)
                    * size_per_allgather
                    * self.world_size
                ],
            )
        self._barrier()

    def test_all_gather_into_tensor_coalesced_async(self):
        self._barrier()
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        num_allgather = 2
        size_per_allgather = 8
        small_in_tensors = [
            torch.ones(size_per_allgather).mlu() for _ in range(num_allgather)
        ]
        small_out_tensors = [
            torch.zeros(size_per_allgather * self.world_size).mlu()
            for _ in range(num_allgather)
        ]
        with dist._coalescing_manager(async_ops=True) as cm:
            for i in range(num_allgather):
                dist.all_gather_into_tensor(small_out_tensors[i], small_in_tensors[i])
        cm.wait()

        big_in_tensor = torch.ones(num_allgather * size_per_allgather).mlu()
        big_out_tensor = torch.zeros(
            num_allgather * size_per_allgather * self.world_size
        ).mlu()
        dist.all_gather_into_tensor(big_out_tensor, big_in_tensor)
        for i in range(num_allgather):
            self.assertEqual(
                small_out_tensors[i],
                big_out_tensor[
                    i
                    * size_per_allgather
                    * self.world_size : (i + 1)
                    * size_per_allgather
                    * self.world_size
                ],
            )
        self._barrier()

    # @unittest.skip("not test")
    def test_abnormal_and_api(self):
        _, rank = self._init_global_test()
        torch.mlu.set_device(rank % torch.mlu.device_count())
        tensors = [self.to_device(torch.randn(2))]
        pg = _get_default_group()

        # test basic api
        self.assertEqual(dist.get_world_size(), int(self.world_size))
        self.assertEqual(dist.get_rank(), self.rank)
        self.assertTrue(dist.is_initialized())

        # test unsupported communicate op
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.allgather_coalesced([tensors], tensors)
        with self.assertRaisesRegex(RuntimeError, "Not supported yet"):
            pg.recv_anysource(tensors, 0)
        # use abnormal input tensors to test
        with self.assertRaisesRegex(
            RuntimeError,
            "Tensor list input to scatter/gather "
            + "must match number of collective participants",
        ):
            pg.allgather([[tensors[0] for _ in range(self.world_size + 1)]], tensors)
        with self.assertRaisesRegex(
            RuntimeError, "Expecting all tensors on the same device"
        ):
            pg.allgather([[tensors[0].cpu() for _ in range(self.world_size)]], tensors)
        with self.assertRaisesRegex(
            RuntimeError,
            "All tensor operands to scatter/gather "
            + "must have the same number of elements",
        ):
            pg.allgather(
                [[tensors[0] for _ in range(self.world_size)]],
                [self.to_device(torch.randn(3))],
            )
        with self.assertRaisesRegex(
            RuntimeError, "Tensors must be on distinct MLU devices"
        ):
            pg.allgather([tensors], [tensors[0], tensors[0]])
        with self.assertRaisesRegex(
            RuntimeError, "Tensors must be on distinct MLU devices"
        ):
            pg.allreduce([tensors[0], tensors[0]])
        with self.assertRaisesRegex(RuntimeError, "Cannot use ReduceOp.BAND with CNCL"):
            pg.allreduce(tensors[0], dist.ReduceOp.BAND)
        with self.assertRaisesRegex(
            RuntimeError, "you passed an empty list of Tensors"
        ):
            pg.broadcast([])
        with self.assertRaisesRegex(
            RuntimeError,
            "Tensor list mustn't be larger than the number of available MLUs",
        ):
            exceed_tensor_list = [
                tensors[0] for _ in range(torch.mlu.device_count() + 1)
            ]
            pg.broadcast(exceed_tensor_list)
        with self.assertRaisesRegex(
            RuntimeError, "No backend type associated with device type cpu"
        ):
            pg.broadcast([tensors[0].cpu()])
        with self.assertRaisesRegex(
            RuntimeError, "Size of input tensor list not equal group size"
        ):
            pg.alltoall(tensors, tensors, dist.AllToAllOptions())
        with self.assertRaisesRegex(RuntimeError, "Tensors must be contiguous"):
            pg.alltoall(
                [tensors[0] for _ in range(self.world_size)],
                [self.to_non_dense(tensors[0]) for _ in range(self.world_size)],
                dist.AllToAllOptions(),
            )
        with self.assertRaisesRegex(
            RuntimeError, "input tensor must be the same type as the output tensor"
        ):
            pg._reduce_scatter_base(tensors[0], tensors[0].half())
        with self.assertRaisesRegex(
            RuntimeError,
            "input tensor must be the same size as output size times world size",
        ):
            pg._reduce_scatter_base(tensors[0], tensors[0])

    # @unittest.skip("not test")
    def test_import_torch_mlu_cncl(self):
        import torch.mlu.cncl


@contextmanager
def _lock():
    TEMP_DIR = os.environ["TEMP_DIR"]
    lockfile = os.path.join(TEMP_DIR, "lockfile")
    with open(lockfile, "w") as lf:
        try:
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_RLCK, 1)
                yield
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            if sys.platform == "win32":
                msvcrt.locking(lf.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
            lf.close()


class Barrier:
    barrier_id = 0

    @classmethod
    def init(cls):
        cls.barrier_id = 0
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        for f_name in os.listdir(barrier_dir):
            os.unlink(os.path.join(barrier_dir, f_name))

    @classmethod
    def sync(cls, wait_for=None, timeout=10):
        if wait_for is None:
            wait_for = dist.get_world_size()
        cls.barrier_id += 1
        barrier_dir = os.path.join(os.environ["TEMP_DIR"], "barrier")
        pid = str(os.getpid())
        barrier_file = os.path.join(barrier_dir, pid)
        with _lock():
            with open(barrier_file, "w") as f:
                f.write(str(cls.barrier_id))

        start_time = time.time()
        while True:
            arrived = 0
            with _lock():
                for f_name in os.listdir(barrier_dir):
                    with open(os.path.join(barrier_dir, f_name)) as f:
                        data = f.read()
                        if int(data) >= cls.barrier_id:
                            arrived += 1
            if arrived == wait_for:
                break

            if time.time() - start_time > timeout:
                raise RuntimeError("barrier timeout")
            time.sleep(0.1)


class TestDistBackend(MultiProcessTestCase, TestCase):
    MANAGER_PROCESS_RANK = -1
    sync_manager = None
    args = None

    @classmethod
    def setUpClass(cls):
        os.environ["MASTER_ADDR"] = cls.args.master_addr
        os.environ["MASTER_PORT"] = str(cls.args.master_port)
        os.environ["WORLD_SIZE"] = str(cls.args.nproc_per_node * cls.args.nnodes)
        os.environ["NPROC_PER_NODE"] = str(cls.args.nproc_per_node)
        os.environ["NNODES"] = str(cls.args.nnodes)
        super().setUpClass()

    def setUp(self):
        # super(TestDistBackend, self).setUp()   # pylint: disable=R1725
        super().setUp()
        # TestCase.setUp(self)
        # initialize temp directories
        initialize_temp_directories()
        # initialize Barrier
        Barrier.init()

    def tearDown(self):
        cleanup_temp_dir()
        super().tearDown()

    @property
    def init_method(self):
        return f"{FILE_SCHEMA}{self.file_name}"

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe):
        self = cls(test_name)
        self.file_name = file_name
        self.rank = rank

        if torch.mlu.device_count() < self.nproc_per_node:
            print("Lack MLU Device !!!!!!")
            sys.exit(0)

        try:
            print("begin init Process", os.getpid())

            dist.init_process_group(
                backend="cncl",
                init_method=self.init_method,
                world_size=int(self.world_size),
                rank=self.rank,
            )
            print("end init Process", os.getpid())
        except RuntimeError as e:
            if "recompile" in e.args[0]:
                sys.exit(SKIP_IF_BACKEND_UNAVAILABLE)
            raise

        self._barrier()
        self.run_test(test_name, pipe)
        self._barrier()
        dist.destroy_process_group()
        sys.exit(0)

    # Needed since MultiProcessTestCase assumes a world_size of 4, but we
    # run these tests under other various world_sizes.
    @property
    def world_size(self):
        return int(os.environ["WORLD_SIZE"])

    @property
    def nproc_per_node(self):
        return int(os.environ["NPROC_PER_NODE"])

    @property
    def nnodes(self):
        return int(os.environ["NNODES"])


class TestDistBackendWithSpawn(TestDistBackend, _DistTestBase):
    def setUp(self):
        super().setUp()
        self._spawn_processes()


if __name__ == "__main__":
    args = parse_args()

    if args.nnodes > 1:
        args.connects = -1

    if args.connects > 0:
        os.environ["CNCL_MLULINK_DISABLE"] = "1"
    if args.connects > 1:
        os.environ["CNCL_P2P_LEVEL"] = "0"
    if args.connects > 2:
        os.environ["CNCL_SHM_DISABLE"] = "1"

    distutils.dir_util.mkpath(TEMP_DIR)

    TestDistBackend.args = args

    run_tests(argv=[sys.argv[0]] + args.unittest_args)
