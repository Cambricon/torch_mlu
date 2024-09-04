from __future__ import print_function
import sys
import os
import copy
import unittest
import logging
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
)

logging.basicConfig(level=logging.DEBUG)


class TestMatmulOp(TestCase):
    # @unittest.skip("not test")
    @testinfo()
    def test_matmul(self):
        mat1_shapes = [
            (8, 9),
            (53, 64, 1024),
            (64, 64, 1024),
            (64, 1024),
            (98, 89),
            (89, 91),
            (20, 0, 1024),
            (30, 0, 512),
            (40, 64, 1024),
            (1024, 0),
            (0, 1024),
            (0, 64),
        ]
        mat2_shapes = [
            (9, 11),
            (1024, 1024),
            (1024, 1024),
            (1024, 94),
            (89, 91),
            (10, 91, 91),
            (1024, 64),
            (512, 0),
            (1024, 0),
            (0, 1024),
            (98, 1024, 0),
            (64, 9),
        ]
        # other type don't be supported in CNNL
        dtypes = [torch.float, torch.double]
        boundary_value = [0.000000001, -0.00001, 999999, -999999]
        for mat1_shape, mat2_shape in zip(mat1_shapes, mat2_shapes):
            for typeId in dtypes:
                mat1 = torch.randn(mat1_shape, dtype=torch.float, requires_grad=True)
                mat2 = torch.randn(mat2_shape, dtype=torch.float, requires_grad=True)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_cpu = torch.matmul(mat1, mat2)
                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                grad = torch.randn(out_cpu.shape)
                out_cpu.backward(grad)
                grad1 = copy.deepcopy(mat1.grad)
                grad2 = copy.deepcopy(mat2.grad)

                mat1.grad.zero_()
                mat2.grad.zero_()

                grad_mlu = self.to_mlu_dtype(grad, typeId)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(grad_mlu)

                self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

                # test boundary value
                in1 = torch.randn(mat1_shape, dtype=torch.float)
                in2 = torch.randn(mat2_shape, dtype=torch.float)
                for bound in boundary_value:
                    mat11 = in1.fill_(bound)
                    mat22 = in2.fill_(bound)

                    # to make mat1 and mat2 as leaf Variables, so in1 and in2 not set
                    # requires_grad
                    mat1 = mat11.clone().requires_grad_(True)
                    mat2 = mat22.clone().requires_grad_(True)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)
                    out_cpu = torch.matmul(mat1, mat2)
                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    grad = torch.randn(out_cpu.shape)
                    out_cpu.backward(grad)
                    grad1 = copy.deepcopy(mat1.grad)
                    grad2 = copy.deepcopy(mat2.grad)

                    mat1.grad.zero_()
                    mat2.grad.zero_()

                    grad_mlu = self.to_mlu_dtype(grad, typeId)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_matmul_not_dense(self):
        mat1_shapes = [
            (8, 9),
            (53, 64, 1024),
            (64, 64, 1024),
            (64, 1024),
            (98, 89),
            (20, 0, 1024),
            (30, 0, 512),
            (40, 64, 1024),
            (1024, 0),
            (0, 64),
        ]
        mat2_shapes = [
            (9, 11),
            (1024, 1024),
            (1024, 1024),
            (1024, 94),
            (89, 91),
            (1024, 64),
            (512, 0),
            (1024, 0),
            (0, 1024),
            (64, 9),
        ]
        # other type don't be supported in CNNL
        dtypes = [torch.float]
        boundary_value = [0.000000001, -0.00001, 999999, -999999]
        for mat1_shape, mat2_shape in zip(mat1_shapes, mat2_shapes):
            for typeId in dtypes:
                mat1_origin = torch.randn(mat1_shape, dtype=torch.float)
                mat2_origin = torch.randn(mat2_shape, dtype=torch.float)
                mat1 = mat1_origin[..., : int(mat1_shape[-1] / 2)].requires_grad_()
                mat2 = mat2_origin[: int(mat2_shape[0] / 2), ...].requires_grad_()
                mat1_mlu = self.to_mlu_dtype(mat1_origin, typeId)[
                    ..., : int(mat1_shape[-1] / 2)
                ].requires_grad_()
                mat2_mlu = self.to_mlu_dtype(mat2_origin, typeId)[
                    : int(mat2_shape[0] / 2), ...
                ].requires_grad_()
                out_cpu = torch.matmul(mat1, mat2)
                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                grad = torch.randn(out_cpu.shape)
                out_cpu.backward(grad)
                grad1 = copy.deepcopy(mat1.grad)
                grad2 = copy.deepcopy(mat2.grad)

                mat1.grad.zero_()
                mat2.grad.zero_()

                grad_mlu = self.to_mlu_dtype(grad, typeId)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(grad_mlu)

                self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

                # test boundary value
                in1 = torch.randn(mat1_shape, dtype=torch.float)
                in2 = torch.randn(mat2_shape, dtype=torch.float)
                for bound in boundary_value:
                    mat11 = in1.fill_(bound)
                    mat22 = in2.fill_(bound)

                    # to make mat1 and mat2 as leaf Variables, so in1 and in2 not set
                    # requires_grad
                    mat1 = mat11.clone().requires_grad_(True)
                    mat2 = mat22.clone().requires_grad_(True)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)
                    out_cpu = torch.matmul(mat1, mat2)
                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    grad = torch.randn(out_cpu.shape)
                    out_cpu.backward(grad)
                    grad1 = copy.deepcopy(mat1.grad)
                    grad2 = copy.deepcopy(mat2.grad)

                    mat1.grad.zero_()
                    mat2.grad.zero_()

                    grad_mlu = self.to_mlu_dtype(grad, typeId)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_matmul_channels_last(self):
        mat1_shapes = [(53, 1, 64, 32), (1, 64, 64, 32), (1, 20, 0, 32)]
        mat2_shapes = [(1, 1, 32, 25), (64, 1, 32, 16), (1, 20, 32, 24)]
        # other type don't be supported in CNNL
        dtypes = [torch.float]
        boundary_value = [0.000000001, -0.00001, 999999, -999999]
        for mat1_shape, mat2_shape in zip(mat1_shapes, mat2_shapes):
            for typeId in dtypes:
                mat1_origin = torch.randn(mat1_shape, dtype=torch.float)
                mat2_origin = torch.randn(mat2_shape, dtype=torch.float)
                mat1 = mat1_origin.to(
                    memory_format=torch.channels_last
                ).requires_grad_()
                mat2 = mat2_origin.to(
                    memory_format=torch.channels_last
                ).requires_grad_()
                mat1_mlu = (
                    self.to_mlu_dtype(mat1_origin, typeId)
                    .to(memory_format=torch.channels_last)
                    .requires_grad_()
                )
                mat2_mlu = (
                    self.to_mlu_dtype(mat2_origin, typeId)
                    .to(memory_format=torch.channels_last)
                    .requires_grad_()
                )
                out_cpu = torch.matmul(mat1, mat2)
                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                grad = torch.randn(out_cpu.shape)
                out_cpu.backward(grad)
                grad1 = copy.deepcopy(mat1.grad)
                grad2 = copy.deepcopy(mat2.grad)

                mat1.grad.zero_()
                mat2.grad.zero_()

                grad_mlu = self.to_mlu_dtype(grad, typeId)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(grad_mlu)

                self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

                # test boundary value
                in1 = torch.randn(mat1_shape, dtype=torch.float)
                in2 = torch.randn(mat2_shape, dtype=torch.float)
                for bound in boundary_value:
                    mat11 = in1.fill_(bound)
                    mat22 = in2.fill_(bound)

                    # to make mat1 and mat2 as leaf Variables, so in1 and in2 not set
                    # requires_grad
                    mat1 = mat11.clone().requires_grad_(True)
                    mat2 = mat22.clone().requires_grad_(True)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)
                    out_cpu = torch.matmul(mat1, mat2)
                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    grad = torch.randn(out_cpu.shape)
                    out_cpu.backward(grad)
                    grad1 = copy.deepcopy(mat1.grad)
                    grad2 = copy.deepcopy(mat2.grad)

                    mat1.grad.zero_()
                    mat2.grad.zero_()

                    grad_mlu = self.to_mlu_dtype(grad, typeId)
                    mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                    mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                    out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                    self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                    out_mlu.backward(grad_mlu)
                    self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                    self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_matmul_exception(self):
        mat1_origin = torch.randn([89, 45], dtype=torch.float)
        mat2_origin = torch.randn([32, 23], dtype=torch.float)
        mat1_mlu = mat1_origin.to("mlu")
        mat2_mlu = mat2_origin.to("mlu")
        ref_msg = r"mat1 and mat2 shapes cannot be multiplied"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            torch.matmul(mat1_mlu, mat2_mlu)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_matmul_large(self):
        mat1_shapes = [(48, 4096, 13725)]
        mat2_shapes = [(13725, 1024)]
        dtypes = [torch.float]
        for mat1_shape, mat2_shape in zip(mat1_shapes, mat2_shapes):
            for typeId in dtypes:
                mat1 = torch.randn(mat1_shape, dtype=torch.float, requires_grad=True)
                mat2 = torch.randn(mat2_shape, dtype=torch.float, requires_grad=True)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_cpu = torch.matmul(mat1, mat2)
                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(
                    out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True
                )

                grad = torch.randn(out_cpu.shape)
                out_cpu.backward(grad)
                grad1 = copy.deepcopy(mat1.grad)
                grad2 = copy.deepcopy(mat2.grad)

                mat1.grad.zero_()
                mat2.grad.zero_()

                grad_mlu = self.to_mlu_dtype(grad, typeId)
                mat1_mlu = self.to_mlu_dtype(mat1, typeId)
                mat2_mlu = self.to_mlu_dtype(mat2, typeId)

                out_mlu = torch.matmul(mat1_mlu, mat2_mlu)
                self.assertTensorsEqual(out_cpu, out_mlu.cpu(), 0.003, use_MSE=True)
                out_mlu.backward(grad_mlu)

                self.assertTensorsEqual(grad1, mat1.grad, 0.003, use_MSE=True)
                self.assertTensorsEqual(grad2, mat2.grad, 0.003, use_MSE=True)

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("48GB")
    def test_matmul_large_exceptions(self):
        # [cnnlMatMul]: the maximum dimension of a should be less than 2147483648(2^31).
        ref_msg = r"CNNL error: CNNL_STATUS_BAD_PARAM"
        with self.assertRaisesRegex(RuntimeError, ref_msg):
            mat1 = torch.randn((1, 4096 * 48 * 13725), dtype=torch.float)
            mat2 = torch.randn((4096 * 48 * 13725, 2), dtype=torch.float)

            out_cpu = torch.matmul(mat1, mat2)
            out_mlu = torch.matmul(mat1.to("mlu"), mat2.to("mlu"))
            self.assertTensorsEqual(out_cpu, out_mlu.cpu().float(), 0.003, use_MSE=True)


if __name__ == "__main__":
    run_tests()
