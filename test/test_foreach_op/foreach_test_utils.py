import torch
import sys
import os
import math
import numpy as np
from typing import cast, Optional, Union
from enum import Enum, auto

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import read_card_info


# This class almost copy from pytorch to check fast path
# https://github.com/pytorch/pytorch/blob/6b0c9c3c6fb46c047f711b82a5033e367b07d2de/test/test_foreach.py#L71
class ForeachFuncWrapper:
    def __init__(self, func):
        self.func = func
        # Some foreach functions don't have in-place implementations.
        self.is_inplace = False if func is None else func.__name__.endswith("_")

    def __call__(self, inputs, **kwargs):
        actual = None
        with torch.profiler.profile() as p:
            actual = self.func(*inputs, **kwargs)
        keys = tuple([e.key for e in p.key_averages()])
        mta_called = any(
            "ForeachBinaryOp" in k
            or "ForeachUnaryOp" in k
            or "ForeachNorm" in k
            or "ForeachLerp" in k
            for k in keys
        )
        assert mta_called == True
        return inputs[0] if self.is_inplace else actual


class ForeachType(Enum):
    UnaryOp = 0
    ReduceOp = auto()
    BinaryOpWithTensor = auto()
    BinaryOpWithScalar = auto()
    BinaryOpWithScalarList = auto()
    BinaryOpWithScalarTensor = auto()
    LerpWithTensor = auto()
    LerpWithScalar = auto()
    BinaryOpWithCPUScalarTensor = auto()


class ForeachOpTest(object):
    def __init__(
        self,
        func,
        foreach_type: ForeachType,
        input_args: Union[tuple, list] = [],
        input_shapes: Union[tuple, list] = [],
        dtypes: Union[tuple, list] = [],
        err: float = 0.003,
        **kwargs,
    ):
        self.cpu_func = func
        self.mlu_func = ForeachFuncWrapper(func)
        self.foreach_type = foreach_type
        # empty inputs cant be handled correctly by cpu foreach_norm
        if foreach_type == ForeachType.ReduceOp and kwargs["ord"] == math.inf:
            self.input_shapes = (
                [
                    (3, 4),
                    (10,),
                    (1, 3, 224, 224),
                    (254, 254, 112, 1, 1, 3),
                ]
                if len(input_shapes) == 0
                else input_shapes
            )
        else:
            self.input_shapes = (
                [
                    (3, 4),
                    (10,),
                    (1, 3, 224, 224),
                    (2, 0, 4),
                    (254, 254, 112, 1, 1, 3),
                    (0, 2, 3),
                ]
                if len(input_shapes) == 0
                else input_shapes
            )
        self.input_args = input_args
        self.err = err
        self.dtypes = (
            [torch.double, torch.float, torch.bfloat16, torch.half]
            if read_card_info() is True
            else [torch.double, torch.float, torch.half]
            if len(dtypes) == 0
            else dtypes
        )
        self.trans_param = (2, 0)
        # Some foreach functions don't have in-place implementations.
        self.is_inplace = False if func is None else func.__name__.endswith("_")
        self.kwargs = kwargs

    def generate_cpu_inputs(self, dtype):
        return [
            torch.testing.make_tensor(shape, dtype=dtype, device="cpu")
            for shape in self.input_shapes
        ]

    def trans_tensors(self, tensors: list):
        return [
            item.transpose(*self.trans_param) if item.dim() >= 3 else item
            for item in tensors
        ]

    def copy_to_mlu(self, tensors: list):
        return [item.mlu() for item in tensors]

    def generate_cpu_and_mlu_tenors(self, dtype):
        tensors = self.generate_cpu_inputs(dtype)
        if self.foreach_type == ForeachType.ReduceOp:
            cpu_tensor = self.trans_tensors([item for item in tensors])
        else:
            cpu_tensor = self.trans_tensors([item.float() for item in tensors])
        mlu_tensors = self.trans_tensors(self.copy_to_mlu(tensors))
        return cpu_tensor, mlu_tensors

    # Foreach op promote all scalar to opmath type.
    def generate_scalar(self):
        value = np.random.randint(-10, 10) * np.random.random(1)[0]
        return value, value

    def generate_scalar_list(self):
        cpu_scalar_list = []
        mlu_scalar_list = []
        for _ in range(len(self.input_shapes)):
            value1, value2 = self.generate_scalar()
            cpu_scalar_list.append(value1)
            mlu_scalar_list.append(value2)
        return cpu_scalar_list, mlu_scalar_list

    def generate_inputs(self, dtype):
        if self.foreach_type == ForeachType.UnaryOp:
            inputs = self.generate_cpu_and_mlu_tenors(dtype)
            return (inputs[0],), (inputs[1],)
        elif self.foreach_type == ForeachType.ReduceOp:
            inputs = self.generate_cpu_and_mlu_tenors(dtype)
            return (inputs[0],), (inputs[1],)
        elif self.foreach_type == ForeachType.BinaryOpWithTensor:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            right_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            return (left_inputs[0], right_inputs[0]), (left_inputs[1], right_inputs[1])
        elif self.foreach_type == ForeachType.BinaryOpWithScalarList:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            right_scalarlist_inputs = self.generate_scalar_list()
            return (left_inputs[0], right_scalarlist_inputs[0]), (
                left_inputs[1],
                right_scalarlist_inputs[1],
            )
        elif self.foreach_type == ForeachType.BinaryOpWithScalar:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            right_scalr_inputs = self.generate_scalar()
            return (left_inputs[0], right_scalr_inputs[0]), (
                left_inputs[1],
                right_scalr_inputs[1],
            )
        elif self.foreach_type == ForeachType.BinaryOpWithScalarTensor:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            cpu_scalar_tensor = torch.testing.make_tensor((), dtype=dtype, device="cpu")
            mlu_scalar_tensor = cpu_scalar_tensor.mlu()
            return (left_inputs[0], cpu_scalar_tensor), (
                left_inputs[1],
                mlu_scalar_tensor,
            )
        elif self.foreach_type == ForeachType.LerpWithTensor:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            right_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            weights = self.generate_cpu_and_mlu_tenors(dtype)
            return (left_inputs[0], right_inputs[0], weights[0]), (
                left_inputs[1],
                right_inputs[1],
                weights[1],
            )
        elif self.foreach_type == ForeachType.LerpWithScalar:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            right_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            weight_scalr_inputs = self.generate_scalar()
            return (left_inputs[0], right_inputs[0], weight_scalr_inputs[0]), (
                left_inputs[1],
                right_inputs[1],
                weight_scalr_inputs[1],
            )
        elif self.foreach_type == ForeachType.BinaryOpWithCPUScalarTensor:
            left_inputs = self.generate_cpu_and_mlu_tenors(dtype)
            cpu_scalar_tensor = torch.testing.make_tensor((), dtype=dtype, device="cpu")
            mlu_scalar_tensor = cpu_scalar_tensor
            return (left_inputs[0], cpu_scalar_tensor), (
                left_inputs[1],
                mlu_scalar_tensor,
            )
        else:
            raise Exception("Invalid ForeachType")

    def __call__(self, value_check, tensor_check):
        for dtype in self.dtypes:
            if dtype == torch.double and self.foreach_type == ForeachType.ReduceOp:
                if self.kwargs["dtype"] == torch.float:
                    # when input dtype is double, output dtype cant be float
                    continue
            cpu_inputs, mlu_inputs = self.generate_inputs(dtype)
            mlu_input_ptr = [item.data_ptr() for item in mlu_inputs[0]]
            out_cpu = self.cpu_func(*cpu_inputs, **self.kwargs)
            if self.is_inplace is True:
                out_cpu = cpu_inputs[0]
            out_mlu = self.mlu_func(mlu_inputs, **self.kwargs)
            for each_cpu, each_mlu, mlu_input_ptr in zip(
                out_cpu, out_mlu, mlu_input_ptr
            ):
                # print(each_cpu, each_mlu)
                if self.is_inplace is True:
                    value_check(each_mlu.data_ptr() == mlu_input_ptr)
                tensor_check(
                    each_cpu.float(), each_mlu.cpu().float(), self.err, use_MSE=True
                )
