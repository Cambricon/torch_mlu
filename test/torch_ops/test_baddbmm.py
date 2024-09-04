import sys
import logging
import os
import copy
import unittest
from random import randint, uniform
from itertools import product
import math
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")
from common_utils import (
    testinfo,
    TestCase,
    TEST_BFLOAT16,
)  # pylint: disable=C0413, C0411

logging.basicConfig(level=logging.DEBUG)

min_shape = 2
max_shape = 16

broadcast_list = [True, False]
zerodim_list = [True, False]
# inf and nan is not supported by cnnl
has_inf_nan_list = [False]
# dtype int is not supported by cnnl
data_type_list = [(torch.float, 3e-3), (torch.half, 3e-3)]
test_list = [broadcast_list, zerodim_list, has_inf_nan_list, data_type_list]


def shape_expand(shape, new_dim):
    """
    expand B,N,M,P with new_dim==0 or 1
    returns a list of [B,N,M,P],[B,N,M,new_dim] ... [new_dim, new_dim, new_dim, new_dim]
    the result can broadcast or zerodims with origin shape
    [[B,N,M,P]] -> [[B,N,M,0], [B,N,M,P], [B,N,0,P] ... ], 16 different shapes.
    """
    all_shapes = [shape]
    for index in range(4):
        tmp_shape = []
        for shapes in all_shapes:
            t_new = copy.deepcopy(shapes)
            t_ori = copy.deepcopy(shapes)
            t_new[index] = new_dim
            tmp_shape.append(copy.deepcopy(t_new))
            tmp_shape.append(copy.deepcopy(t_ori))
        all_shapes = copy.deepcopy(tmp_shape)
        # for X from B to P:
        # all_shapes = [..., [B,N,X,P], ...] -> [..., [B,N,new_dim,P], [B,N,X,P], ...]
    return all_shapes


def shape_generator(broadcast=False, zerodim=False):
    """
    Returns all lists of tuples as shapes:
    [(shapeInput, shapeMat1, shapeMat2), (...), ...]
    ShapeInput is (B,N,P), shapeMat1 is (B,N,M) and shapeMat2 is (B,M,P).
    When broadcast or zerodim is set as True, shapes like (B, 1, P) etc will add to lists.
    """

    # not support zero dim and broadcast at same time
    assert broadcast is False or zerodim is False

    basic_shape = [
        randint(min_shape, max_shape) for i in range(4)
    ]  # basic_shape is BNMP
    B, N, M, P = basic_shape
    if not broadcast and not zerodim:
        return [((B, N, P), (B, N, M), (B, M, P))]

    if zerodim:
        all_shapes = shape_expand(basic_shape, 0)
        result_shapes = []
        for B, N, M, P in all_shapes:
            result_shapes.append(((B, N, P), (B, N, M), (B, M, P)))
        assert len(result_shapes) == 16
        return result_shapes

    if broadcast:
        all_shapes = shape_expand(basic_shape, 1)
        result_shapes = []
        for b, n, m, p in all_shapes:
            # Add input broad cast:
            if m != 1:
                continue  # skip half of m as input only use bnp
            result_shapes.append(((b, n, p), (B, N, M), (B, M, P)))
        assert len(result_shapes) == 8
        return result_shapes
    return []


def add_inf_nan(tensor, has_inf_nan=False):
    """
    generator tensor with or without inf and(or) nan
    """
    inf = math.inf
    nan = math.nan
    assert tensor.device == torch.device("cpu")

    tmp_tensor = tensor.view(-1)
    tensor_num = tmp_tensor.numel()
    if tensor_num == 0:
        return tensor

    inf_nan_num = randint(0, tensor_num // 3)
    if has_inf_nan:
        for _ in range(inf_nan_num):
            tmp_tensor[randint(0, tensor_num - 1)] = inf
            tmp_tensor[randint(0, tensor_num - 1)] = -inf
            tmp_tensor[randint(0, tensor_num - 1)] = nan
    return tmp_tensor.view(tensor.size())


class TestBaddbmm(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm(self):
        convert_func = [self.to_non_dense, self.convert_to_channel_last, lambda x: x]
        for convert1, convert2, convert3 in product(*([convert_func] * 3)):
            for broadcast, zerodim, has_inf_nan, dtype_err in product(*test_list):
                if broadcast and zerodim:
                    continue
                dtype = dtype_err[0]
                if (dtype == torch.int) and (has_inf_nan):
                    # nan and inf only for float
                    continue
                err = dtype_err[1]
                shape_table = shape_generator(broadcast, zerodim)
                for input_shape, mat1_shape, mat2_shape in shape_table:
                    input_cpu = add_inf_nan(
                        torch.randn(input_shape).to(dtype), has_inf_nan
                    )
                    mat1_cpu = torch.randn(mat1_shape).to(dtype)
                    mat2_cpu = torch.randn(mat2_shape).to(dtype)

                    alpha = uniform(-10, 10)
                    beta = uniform(-10, 10)
                    if has_inf_nan:
                        beta = 0
                    if dtype == torch.int:
                        alpha = int(alpha)
                        beta = int(beta)
                    input_mlu = convert1(copy.deepcopy(input_cpu).to("mlu"))
                    mat1_mlu = convert2(copy.deepcopy(mat1_cpu).to("mlu"))
                    mat2_mlu = convert3(copy.deepcopy(mat2_cpu).to("mlu"))

                    if dtype == torch.half:
                        input_cpu = input_cpu.to(torch.float)
                        mat1_cpu = mat1_cpu.to(torch.float)
                        mat2_cpu = mat2_cpu.to(torch.float)

                    result_cpu = torch.baddbmm(
                        input_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha
                    )
                    result_mlu = torch.baddbmm(
                        input_mlu, mat1_mlu, mat2_mlu, beta=beta, alpha=alpha
                    )
                    self.assertTensorsEqual(
                        result_cpu.float(),
                        result_mlu.cpu().contiguous().float(),
                        err,
                        use_MSE=True,
                    )

    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm_beta_not_zero(self):
        alpha = 0
        beta = 1
        input_cpu = torch.randn((10, 12, 5))
        batch1 = torch.randn((10, 12, 8))
        batch2 = torch.randn((10, 8, 5))
        out_cpu = torch.baddbmm(input_cpu, batch1, batch2, beta=beta, alpha=alpha)
        out_device = torch.baddbmm(
            input_cpu.to("mlu"),
            batch1.to("mlu"),
            batch2.to("mlu"),
            beta=beta,
            alpha=alpha,
        )
        self.assertTensorsEqual(
            out_cpu.float(), out_device.cpu().contiguous().float(), 3e-3, use_MSE=True
        )

    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm_inplace(self):
        convert_func = [self.to_non_dense, self.convert_to_channel_last, lambda x: x]
        for convert1, convert2, convert3 in product(*([convert_func] * 3)):
            for broadcast, zerodim, has_inf_nan, dtype_err in product(*test_list):
                if broadcast:  # inplace calculation is not broadcastable of input
                    continue
                dtype = dtype_err[0]
                if (dtype == torch.int) and (has_inf_nan):
                    continue
                err = dtype_err[1]
                shape_table = shape_generator(broadcast, zerodim)
                for input_shape, mat1_shape, mat2_shape in shape_table:
                    input_cpu = add_inf_nan(
                        torch.randn(input_shape).to(dtype), has_inf_nan
                    )
                    mat1_cpu = torch.randn(mat1_shape).to(dtype)
                    mat2_cpu = torch.randn(mat2_shape).to(dtype)

                    alpha = uniform(-10, 10)
                    beta = uniform(-10, 10)

                    if has_inf_nan:
                        beta = 0
                    if dtype == torch.int:
                        alpha = int(alpha)
                        beta = int(beta)
                    input_mlu = convert1(copy.deepcopy(input_cpu).to("mlu"))
                    mat1_mlu = convert2(copy.deepcopy(mat1_cpu).to("mlu"))
                    mat2_mlu = convert3(copy.deepcopy(mat2_cpu).to("mlu"))
                    if dtype == torch.half:
                        input_cpu = input_cpu.to(torch.float)
                        mat1_cpu = mat1_cpu.to(torch.float)
                        mat2_cpu = mat2_cpu.to(torch.float)
                    input_mlu_ptr = input_mlu.data_ptr()
                    input_cpu.baddbmm_(mat1_cpu, mat2_cpu, beta=beta, alpha=alpha)
                    input_mlu.baddbmm_(mat1_mlu, mat2_mlu, beta=beta, alpha=alpha)
                    self.assertTensorsEqual(
                        input_cpu.float(),
                        input_mlu.cpu().contiguous().float(),
                        err,
                        use_MSE=True,
                    )
                    self.assertEqual(input_mlu_ptr, input_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm_out(self):
        convert_func = [self.to_non_dense, self.convert_to_channel_last, lambda x: x]
        for convert1, convert2, convert3 in product(*([convert_func] * 3)):
            for broadcast, zerodim, has_inf_nan, dtype_err in product(*test_list):
                if broadcast and zerodim:
                    continue
                dtype = dtype_err[0]
                if (dtype == torch.int) and (has_inf_nan):
                    continue
                err = dtype_err[1]
                shape_table = shape_generator(broadcast, zerodim)
                for input_shape, mat1_shape, mat2_shape in shape_table:
                    input_cpu = add_inf_nan(
                        torch.randn(input_shape).to(dtype), has_inf_nan
                    )
                    mat1_cpu = torch.randn(mat1_shape).to(dtype)
                    mat2_cpu = torch.randn(mat2_shape).to(dtype)

                    alpha = uniform(-10, 10)
                    beta = uniform(-10, 10)
                    if has_inf_nan:
                        beta = 0
                    if dtype == torch.int:
                        alpha = int(alpha)
                        beta = int(beta)

                    input_mlu = convert1(copy.deepcopy(input_cpu).to("mlu"))
                    mat1_mlu = convert2(copy.deepcopy(mat1_cpu).to("mlu"))
                    mat2_mlu = convert3(copy.deepcopy(mat2_cpu).to("mlu"))

                    if dtype == torch.half:
                        input_cpu = input_cpu.to(torch.float)
                        mat1_cpu = mat1_cpu.to(torch.float)
                        mat2_cpu = mat2_cpu.to(torch.float)

                    result_cpu = torch.baddbmm(
                        input_cpu, mat1_cpu, mat2_cpu, beta=beta, alpha=alpha
                    )
                    torch.baddbmm(
                        input_cpu,
                        mat1_cpu,
                        mat2_cpu,
                        beta=beta,
                        alpha=alpha,
                        out=result_cpu,
                    )
                    result_mlu = (
                        torch.randn(result_cpu.size()).to(input_mlu.dtype).to("mlu")
                    )
                    result_mlu_ptr = result_mlu.data_ptr()
                    torch.baddbmm(
                        input_mlu,
                        mat1_mlu,
                        mat2_mlu,
                        beta=beta,
                        alpha=alpha,
                        out=result_mlu,
                    )
                    self.assertTensorsEqual(
                        result_cpu.float(),
                        result_mlu.cpu().contiguous().float(),
                        err,
                        use_MSE=True,
                    )
                    self.assertEqual(result_mlu_ptr, result_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm_exceptions(self):
        input = torch.randn((8, 4, 6), dtype=torch.float)
        batch1 = torch.randn((8, 4, 5), dtype=torch.half)
        batch2 = torch.randn((8, 5, 6), dtype=torch.float)
        ref_msg = "Input dtypes must be the same, got: input float, batch1: c10::Half, batch2: float"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.baddbmm(input.mlu(), batch1.mlu(), batch2.mlu(), beta=1.1)

        input = torch.randn((8, 4, 6)).int()
        batch1 = torch.randn((8, 4, 5)).int()
        batch2 = torch.randn((8, 5, 6)).int()
        ref_msg = f"not implemented for 'Int'"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.baddbmm(input.mlu(), batch1.mlu(), batch2.mlu(), beta=2)

    # @unittest.skip("not test")
    @testinfo()
    def test_baddbmm_permute(self):
        input = torch.randn((8, 4, 6), dtype=torch.float)
        batch1 = torch.randn((4, 8, 5), dtype=torch.float)
        batch2 = torch.randn((5, 8, 6), dtype=torch.float)
        out_cpu = torch.baddbmm(
            input, batch1.permute(1, 0, 2), batch2.permute(1, 0, 2), beta=1.1
        )
        out_mlu = torch.baddbmm(
            input.mlu(),
            batch1.mlu().permute(1, 0, 2),
            batch2.mlu().permute(1, 0, 2),
            beta=1.1,
        )
        self.assertTensorsEqual(out_cpu, out_mlu.cpu().contiguous(), 3e-3, use_MSE=True)

    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_baddbmm_bfloat16(self):
        # baddbmm is same with bmm.
        alpha = 1
        beta = 1
        input = torch.randn((10, 12, 5), dtype=torch.bfloat16).float()
        batch1 = torch.randn((10, 12, 8), dtype=torch.bfloat16).float()
        batch2 = torch.randn((10, 8, 5), dtype=torch.bfloat16).float()
        input_cpu = torch.nn.Parameter(input)
        batch1_cpu = torch.nn.Parameter(batch1)
        batch2_cpu = torch.nn.Parameter(batch2)
        input_mlu = torch.nn.Parameter(input.mlu().bfloat16())
        batch1_mlu = torch.nn.Parameter(batch1.mlu().bfloat16())
        batch2_mlu = torch.nn.Parameter(batch2.mlu().bfloat16())
        out_cpu = torch.baddbmm(
            input_cpu, batch1_cpu, batch2_cpu, beta=beta, alpha=alpha
        )
        out_device = torch.baddbmm(
            input_mlu, batch1_mlu, batch2_mlu, beta=beta, alpha=alpha
        )
        grad = torch.randn(out_cpu.shape).bfloat16().float()
        grad_mlu = grad.mlu().bfloat16()
        out_cpu.backward(grad)
        out_device.backward(grad_mlu)
        self.assertTensorsEqual(
            out_cpu.float(), out_device.cpu().float(), 3e-3, use_MSE=True
        )
        self.assertTensorsEqual(
            input_cpu.grad.float(), input_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            batch1_cpu.grad.float(), batch1_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )
        self.assertTensorsEqual(
            batch2_cpu.grad.float(), batch2_mlu.grad.cpu().float(), 0.003, use_MSE=True
        )


if __name__ == "__main__":
    unittest.main()
