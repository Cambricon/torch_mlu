from __future__ import print_function

import sys
import os
import time
import unittest
import logging
from functools import partial
from itertools import product
from enum import Enum
import numpy as np
import torch

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import testinfo, TestCase  # pylint: disable=C0413,C0411

logging.basicConfig(level=logging.DEBUG)

CNNL_DIM_MAX = 8


# Get random expand parameters
# axis_num is mean expand dims num, like input ndim is 3, axis_num is 2,
# then output tensor ndim is 5.
def get_expand_parameter(input: torch.Tensor, axis_num: int):
    tensor_dim = input.dim()
    if (tensor_dim + axis_num) >= CNNL_DIM_MAX:
        axis_num = CNNL_DIM_MAX - tensor_dim
    expand_parameter = list(input.size())
    max_dim_value = max(expand_parameter) + 2
    expand_dim_num = 0
    for i in range(tensor_dim):
        if expand_parameter[i] == 1:
            expand_parameter[i] = np.random.randint(1, max_dim_value)
            expand_dim_num += 1
    while expand_dim_num < axis_num:
        expand_parameter.insert(0, expand_dim_num + 1)
        expand_dim_num += 1
    return expand_parameter


# Get random permute parameters
def get_permute_parameter(input: torch.Tensor):
    dim_vec = np.arange(input.dim())
    np.random.shuffle(dim_vec)
    dim_vec = list(dim_vec)
    return dim_vec


# Get random slice parameter, now using select instand slice,
# There are almost equal in view op node. select is combined by
# slice and squeeze.
def get_select_parameter(input: torch.Tensor):
    tensor_size = list(input.size())
    dim_value = 1
    flag = 0
    if len(tensor_size) == 0:
        return 0, 0
    dim = np.random.randint(0, len(tensor_size))
    # Get dim which dim value is greater than 1.
    while dim_value <= 1 and flag < len(tensor_size):
        dim = 0 if dim == len(tensor_size) - 1 else dim + 1
        dim_value = tensor_size[dim]  # pylint: disable=E1126
        flag += 1
    if dim_value == 1:
        return 0, 0
    index = np.random.randint(1, dim_value)
    return dim, index


# reshapeFormat enum is used to control reshape behavior
# Split is mean split one dimension to two dimension
class reshapeFormat(Enum):
    Split = 0
    Merge = 1
    Insert = 2
    Eliminate = 3


# Get random reshape parameter, only create safe view parameter.
# Note: unsafe view will trigger copy and interrupt view chain.
def get_reshape_parameter(input: torch.Tensor):  # pylint: disable=C0301,R0912
    tensor_size = list(input.size())
    reshape_format = np.random.randint(
        reshapeFormat.Split.value, reshapeFormat.Eliminate.value + 1
    )
    if len(tensor_size) in (0, 1):
        reshape_format = reshapeFormat.Insert.value
        dim = 0
    elif len(tensor_size) >= CNNL_DIM_MAX:
        reshape_format = reshapeFormat.Eliminate.value
        dim = np.random.randint(0, CNNL_DIM_MAX)
    else:
        dim = np.random.randint(0, len(tensor_size))

    if reshape_format == reshapeFormat.Split.value:
        # Get simple split value.
        if len(tensor_size) == CNNL_DIM_MAX:
            return tensor_size
        base_split_value = 0
        for i in range(2, 6):
            if (tensor_size[dim] % i) == 0:
                base_split_value = i
                break
        if base_split_value != 0:
            tensor_size[dim] = int(tensor_size[dim] / base_split_value)
            tensor_size.insert(dim, base_split_value)
        return tensor_size
    elif reshape_format == reshapeFormat.Merge.value:
        # Only merge forward dim, expect dim value is equal to 0.
        another_dim = dim + 1 if dim == 0 else dim - 1
        tensor_size[dim] *= tensor_size[another_dim]
        tensor_size.pop(another_dim)
        return tensor_size
    elif reshape_format == reshapeFormat.Insert.value:
        # Only insert value 1 to tensor size.
        insert_num = np.random.randint(0, 3)
        insert_num = (
            CNNL_DIM_MAX - len(tensor_size)
            if len(tensor_size) + insert_num > CNNL_DIM_MAX
            else insert_num
        )
        for i in range(insert_num):
            tensor_size.insert(dim, 1)
        return tensor_size
    else:
        # Only pop size value 1 in tensor size.
        pop_num = np.random.randint(0, 3)
        remove_value = 1
        for i in range(pop_num):
            if remove_value in tensor_size:
                tensor_size.remove(remove_value)
        return tensor_size


# Get random squeeze or unsqueeze output
def squeeze_or_unsqueeze_tensor(input: torch.Tensor):
    one_dim = [index for index, value in enumerate(input.size()) if value == 1]
    # add random to is_squeeze by using np random.
    is_squeeze = len(one_dim) != 0 and np.random.randint(0, 2) == 0
    # Using random index of squeeze or unsqueeze dim.
    dim_index = (
        np.random.randint(0, len(one_dim))
        if is_squeeze
        else np.random.randint(0, input.dim())
    )
    return (
        input.squeeze(one_dim[dim_index]) if is_squeeze else input.unsqueeze(dim_index)
    )


# Get unfold op random parameters.
def get_unfold_parameter(input: torch.Tensor):
    dim_size = 0
    while dim_size <= 1:
        dim = np.random.randint(0, input.dim())
        dim_size = input.size(dim)
    size = np.random.randint(1, dim_size)
    step = np.random.randint(1, size) if size > 1 else 1
    return dim, size, step


# permute + expand + slice
class ViewChain1(torch.nn.Module):
    def __init__(self):
        super(ViewChain1, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        axis_num = np.random.randint(0, 2)
        v_expand = get_expand_parameter(x, axis_num)
        x = x.expand(v_expand)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        return x + 1


# expand + permute + slice
class ViewChain2(torch.nn.Module):
    def __init__(self):
        super(ViewChain2, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        axis_num = np.random.randint(0, 2)
        v_expand = get_expand_parameter(x, axis_num)
        x = x.expand(v_expand)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        return x + 1


# slice + expand + permute
class ViewChain3(torch.nn.Module):
    def __init__(self):
        super(ViewChain3, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        axis_num = np.random.randint(0, 2)
        v_expand = get_expand_parameter(x, axis_num)
        x = x.expand(v_expand)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        return x + 1


# slice + permute + permute + expand
class ViewChain4(torch.nn.Module):
    def __init__(self):
        super(ViewChain4, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        axis_num = np.random.randint(0, 2)
        v_expand = get_expand_parameter(x, axis_num)
        x = x.expand(v_expand)
        return x + 1


# slice + permute + permute + reshape + reshape
class ViewChain5(torch.nn.Module):
    def __init__(self):
        super(ViewChain5, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        return x + 1


# slice + reshape + slice + reshape + permute + reshape
# + reshape + permute + slice + reshape + reshape + slice
# + reshape
class ViewChain6(torch.nn.Module):
    def __init__(self):
        super(ViewChain6, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        return x + 1


# slice + permute + permute + reshape + reshape + permute
# + reshape + permute + reshape
class ViewChain7(torch.nn.Module):
    def __init__(self):
        super(ViewChain7, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_reshape = get_reshape_parameter(x)
        x = x.reshape(v_reshape)
        return x + 1


# permute + squeeze(or unsqueeze) + permute + select
class ViewChain8(torch.nn.Module):
    def __init__(self):
        super(ViewChain8, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        x = squeeze_or_unsqueeze_tensor(x)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        dim, index = get_select_parameter(x)
        if x.dim() != 0:
            x = x.select(dim, index)
        return x + 1


# permute + squeeze(or unsqueeze) + unfold + permute + permute + unfold + select
# + permute + unfold + select
class ViewChain9(torch.nn.Module):
    def __init__(self):
        super(ViewChain9, self).__init__()

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x need be a torch tensor."
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        x = squeeze_or_unsqueeze_tensor(x)
        dim, size, step = get_unfold_parameter(x)
        x = x.unfold(dim, size, step)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        v_permute = get_permute_parameter(x)
        x = x.permute(v_permute)
        dim, size, step = get_unfold_parameter(x)
        x = x.unfold(dim, size, step)
        # dim, index = get_select_parameter(x)
        # if x.dim() != 0:
        #    x = x.select(dim, index)
        return x + 1


class TestOps(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_view_chain(self):
        shape_list = [
            (3, 224, 224),
            (2, 3, 224, 224),
            (2, 1, 224, 224),
            (2, 10, 1, 24, 24),
            (2, 10, 3, 24, 24),
        ]
        func_list = [
            self.convert_to_channel_last,
            self.to_non_dense,
            self.convert_to_is_non_overlapping_and_dense,
            lambda x: x,
        ]
        view_list = [
            ViewChain1,
            ViewChain2,
            ViewChain3,
            ViewChain4,
            ViewChain5,
            ViewChain6,
            ViewChain7,
            ViewChain8,
            ViewChain9,
        ]
        for shape, view_func, func in product(shape_list, view_list, func_list):
            seed = int(time.time())
            viewchain = view_func()
            input_t = torch.rand(shape)
            input_mlu = input_t.to("mlu")
            if func.__name__ == "convert_to_is_non_overlapping_and_dense":
                func_partial = partial(func, seed=seed)
            else:
                func_partial = func
            np.random.seed(seed)
            output_cpu = viewchain(func_partial(input_t))
            np.random.seed(seed)
            output_mlu = viewchain(func_partial(input_mlu))
            msg = "seed: " + str(seed)
            self.assertTensorsEqual(
                output_cpu, output_mlu.cpu(), 0, msg, use_MSE=True
            )  # pylint: disable=C0209

    # Add permute + reshape + permute view chain optimization testcase.
    # Cause view chain test is using random view op sequeeze which may not contained this
    # format.
    # @unittest.skip("not test")
    @testinfo()
    def test_permute_squeeze_permute(self):
        shape_permute = [[(3, 24, 1, 24), (0, 2, 3, 1), 1, (2, 0, 1)]]
        for shape, permute_index1, squeeze_dim, permute_index2 in shape_permute:
            input_t = torch.rand(shape)
            input_mlu = input_t.mlu()
            output_cpu = (
                input_t[:, 0:20:2, ...]
                .permute(permute_index1)
                .squeeze(squeeze_dim)
                .permute(permute_index2)
            )
            output_mlu = (
                input_mlu[:, 0:20:2, ...]
                .permute(permute_index1)
                .squeeze(squeeze_dim)
                .permute(permute_index2)
            )
            self.assertTensorsEqual(output_cpu, output_mlu.cpu(), 0, use_MSE=True)


if __name__ == "__main__":
    unittest.main()
