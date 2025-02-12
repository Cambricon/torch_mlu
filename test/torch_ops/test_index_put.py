from __future__ import print_function

import sys
import os
import unittest
import logging
import copy

import numpy as np
import torch
import torch_mlu

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../")

# pylint: disable=C0413
from common_utils import (
    testinfo,
    run_tests,
    TestCase,
    TEST_LARGETENSOR,
    largeTensorTest,
    read_card_info,
)

TEST_BFLOAT16 = read_card_info()
logging.basicConfig(level=logging.DEBUG)


class TestIndexPut(TestCase):
    def _generate_tensor(self, shape, dtype, value=16777216, reduce_dim=False):
        if dtype == torch.bool:
            out = torch.randint(2, shape).type(dtype)
        else:
            # TODO(liuyuxin): test negative number in future.
            out = torch.randint(value, shape).type(dtype)
        if len(shape) == 1 and reduce_dim:
            out = out[0]
        return out

    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_network(self):
        param_list = [
            ((4, 30000), [(4,), (4,)], torch.long, torch.randn(4)),
            ((1, 3, 2560), [(1, 3)], torch.bool, 1.1),
            ((2048,), [(2048,)], torch.bool, -1),
            ((8732,), [(1,)], torch.long, -1),
            ((8732,), [(0,)], torch.long, -1),
            ((0,), [(0,)], torch.long, -1),
            ((55, 555), [(55, 555)], torch.bool, 1.1),
            ((8, 8732), [(8, 8732)], torch.bool, 1.1),
            ((16, 16, 16), [(4, 3)], torch.long, 100),
            ((512, 256, 7, 7), [(0,)], torch.long, torch.randn([0, 256, 7, 7])),
            ((1, 1, 2560), [(1, 1)], torch.bool, 1.1),
        ]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = self._generate_tensor(input_shape, torch.float)
            input_mlu = self.to_device(input)
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                if input_shape[i] == 0:
                    indice = self._generate_tensor(indices_shape[i], type_)
                else:
                    indice = self._generate_tensor(
                        indices_shape[i], type_, input_shape[i]
                    )
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            input[indices] = value_
            input_mlu[indices_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((4, 2), (1,), torch.long, torch.randn(2))]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[0])
            indice_mlu = self.to_device(indice)
            input[indice] = value_
            input_mlu[indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # with undefined indices
        param_list = [((2, 2), (1,), torch.long, torch.randn(2, 1))]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[1])
            indice_mlu = self.to_device(indice)
            input[:, indice] = value_
            input_mlu[:, indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((2, 2), (1,), torch.long, 1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[0])
            indice_mlu = self.to_device(indice)
            input[indice, :] = value_
            input_mlu[indice_mlu, :] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        param_list = [((32, 3, 40, 84, 85), (2,), torch.long, 1)]
        for input_shape, indices_shape, type_, value_ in param_list:
            input = torch.randn(input_shape)
            input_mlu = self.to_device(input)
            indice = self._generate_tensor(indices_shape, type_, input_shape[4])
            indice_mlu = self.to_device(indice)
            input[..., indice] = value_
            input_mlu[..., indice_mlu] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # with multiple indices
        # repeatedly writing-in may cause unexpected error,
        # so make sure writing with the same value into same place.
        param_list = [
            ((16, 3, 13, 13), (132,), torch.float, torch.long, torch.randn(132)),
            (
                (16, 3, 13, 13),
                (132,),
                torch.long,
                torch.long,
                torch.randn(132).to(torch.long),
            ),
        ]
        for input_shape, indices_shape, input_dtype, index_dtype, value_ in param_list:
            input = torch.randn(input_shape).to(input_dtype)
            input_mlu = self.to_device(input)
            repeat_num = indices_shape[0] // input_shape[1]
            indice_1 = torch.randperm(input_shape[1], dtype=index_dtype).repeat(
                repeat_num
            )
            indice_2 = torch.randperm(input_shape[1], dtype=index_dtype).repeat(
                repeat_num
            )
            indice_3 = torch.randperm(input_shape[1], dtype=index_dtype).repeat(
                repeat_num
            )
            indice_4 = torch.randperm(input_shape[1], dtype=index_dtype).repeat(
                repeat_num
            )
            value_ = torch.randn(input_shape[1]).repeat(repeat_num).to(input_dtype)
            input[indice_1, indice_2, indice_3, indice_4] = value_
            input_mlu[
                self.to_device(indice_1),
                self.to_device(indice_2),
                self.to_device(indice_3),
                self.to_device(indice_4),
            ] = self.to_device(value_)
            self.assertTensorsEqual(input, input_mlu.cpu(), 0.00, use_MSE=True)

        # accumulate = True
        param_list = [
            ((8, 8732, 4), [(8, 8732, 4)], torch.float, torch.bool, 1),
            ((8, 8732, 4), [(8, 8732, 4)], torch.long, torch.long, 1),
        ]
        for input_shape, indices_shape, input_dtype, index_dtype, value_ in param_list:
            input = torch.randn(input_shape).to(input_dtype)
            input_mlu = self.to_device(input)
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                indice = self._generate_tensor(
                    indices_shape[i], index_dtype, input_shape[i]
                )
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            output = torch.index_put(
                input, indices, torch.tensor(value_).to(input_dtype), True
            )
            output_mlu = torch.index_put(
                input_mlu,
                indices_mlu,
                self.to_device(torch.tensor(value_).to(input_dtype)),
                True,
            )
            self.assertTensorsEqual(output, output_mlu.cpu(), 0.00, use_MSE=True)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_not_contiguous(self):
        shape_index = [
            ((8, 6), [1, 3, 6]),
            ((13, 15, 14), [1, 3, 5, 8]),
            ((53, 71, 3, 3), [2, 4, 6, 7, 8]),
        ]
        for shape, index in shape_index:
            x = torch.randn(shape).float()
            x_mlu = x.to("mlu")
            x_mlu_ptr = x_mlu.data_ptr()
            indices = torch.tensor(index).long()
            indices_mlu = indices.to("mlu")
            value = torch.randn(x[indices, :5].shape).float()
            value_mlu = value.to("mlu")
            x[indices, :5] = value
            x_mlu[indices_mlu, :5] = value_mlu
            self.assertTensorsEqual(x, x_mlu.cpu(), 0.00)
            self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())

    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_empty_indices(self):
        x = torch.randn(2, 2, 2).bool()
        x_mlu = x.mlu()
        x_mlu_ptr = x_mlu.data_ptr()
        indices = (
            torch.tensor([], dtype=torch.int64, device="cpu"),
            torch.tensor([], dtype=torch.int64, device="cpu"),
        )
        indices_mlu = (
            torch.tensor([], dtype=torch.int64, device="mlu"),
            torch.tensor([], dtype=torch.int64, device="mlu"),
        )
        x[indices] = False
        x_mlu[indices_mlu] = False
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.00)
        self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
        x[[]] = False
        x_mlu[indices_mlu] = False
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.00)
        self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())
        # The following case is provided by PYTORCH-8245
        reference = torch.arange(0.0, 160, dtype=torch.long, device="cpu").view(4, 8, 5)
        reference_mlu = reference.mlu()
        reference_mlu_ptr = reference_mlu.data_ptr()
        indexer = [[0, 2, 3], slice(None), [1, 3, 4]]
        reference[indexer] = 212
        reference_mlu[indexer] = 212
        self.assertTensorsEqual(reference, reference_mlu.cpu(), 0.00)
        self.assertEqual(reference_mlu_ptr, reference_mlu.data_ptr())

    # The following test cases are used to test whether the index_put works correctly with an empty input
    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_empty_input(self):
        a = torch.tensor([])
        a_mlu = a.mlu()
        a_mlu_ptr = a_mlu.data_ptr()
        empty_indice = torch.tensor([]).long()
        empty_indice_mlu = torch.tensor([]).mlu().long()
        value_long = torch.tensor([1])
        value_long_mlu = value_long.mlu()
        # The following 4 cases are used to test whether correct exceptions are thrown when illegal
        # inputs are given (index with wrong type, scalar & tensor indices, values with mismatch dtype)
        ref_msg_0 = "tensors used as indices must be long, byte or bool tensors"
        with self.assertRaises(IndexError) as info:
            a_mlu[torch.tensor([]).mlu()] = 1
        self.assertEqual(info.exception.args[0], ref_msg_0)
        ref_msg_1 = "index 1 is out of bounds for dimension 0 with size 0"
        with self.assertRaises(IndexError) as info:
            a_mlu[1] = 0
        self.assertEqual(info.exception.args[0], ref_msg_1)
        ref_msg_2 = "index is out of bounds for dimension with size 0"
        with self.assertRaises(IndexError) as info:
            a_mlu[torch.tensor([1]).mlu().long()] = 0
        self.assertEqual(info.exception.args[0], ref_msg_2)
        ref_msg_3 = (
            "Index put requires the source and destination dtypes match, "
            "got Float for the destination and Long for the source."
        )
        with self.assertRaises(RuntimeError) as info:
            a_mlu[empty_indice_mlu] = value_long_mlu
        self.assertEqual(info.exception.args[0], ref_msg_3)
        # The following 2 cases are used to test whether the index_put op returns correct results
        # when legal inputs are given (empty indices, scalar & Tensor values)
        a[empty_indice] = 0
        a_mlu[empty_indice_mlu] = 0
        self.assertTensorsEqual(a, a_mlu.cpu(), 0.00)
        self.assertEqual(a_mlu_ptr, a_mlu.data_ptr())
        a[empty_indice] = value_long.float()
        a_mlu[empty_indice_mlu] = value_long_mlu.float()
        self.assertTensorsEqual(a, a_mlu.cpu(), 0.00)
        self.assertEqual(a_mlu_ptr, a_mlu.data_ptr())

    # This test case is used to ensure that the index_put op gives correct result when there is
    # non-contiguous subspace in indices and self, please refer PYTORCH-8439 and PYTORCH-8525 for
    # more details.
    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_high_dim_backward(self):
        input = torch.nn.parameter.Parameter(
            torch.rand(16, 3, 80, 20, 20), requires_grad=True
        )
        input_mlu = input.mlu()
        index1 = torch.randint(size=(100,), high=16).mlu()
        index2 = torch.randint(size=(100,), high=3).mlu()
        index3 = torch.randint(size=(100,), high=20).mlu()
        index4 = torch.randint(size=(100,), high=20).mlu()
        r = input[index1.cpu(), index2.cpu(), :, index3.cpu(), index4.cpu()]
        l = r.sum()
        r_mlu = input_mlu[index1, index2, :, index3, index4]
        l_mlu = r_mlu.sum()
        grad = torch.randn(l.shape)
        grad_mlu = copy.deepcopy(grad).to("mlu")
        l.backward(grad)
        l_mlu.backward(grad_mlu)
        self.assertTensorsEqual(l, l_mlu.cpu(), 0.003)

    # @unittest.skip("not test")
    @testinfo()
    def test_index_put_exception(self):
        a = self.to_device(torch.rand(2, 3, 4))
        indice1 = self.to_device(torch.tensor([0, 1, 0]))
        indice2 = torch.tensor([0, 1])
        val = torch.randn(1)
        ref_msg_0 = r"^indices have more indices than self dim"
        with self.assertRaisesRegex(RuntimeError, ref_msg_0):
            a.index_put_(
                (indice1, self.to_device(a[0, :, :] > 0), indice1, indice1),
                values=self.to_device(val),
            )
        ref_msg_1 = "shape mismatch: indexing tensors could not be broadcast together with shapes [3], [12], [12]"
        with self.assertRaises(IndexError) as info:
            a.index_put_(
                (indice1, self.to_device(a[0, :, :] > 0)),
                values=self.to_device(val.int()),
            )
        self.assertEqual(info.exception.args[0], ref_msg_1)
        ref_msg_2 = "tensors used as indices must be long, byte or bool tensors"
        with self.assertRaisesRegex(IndexError, ref_msg_2):
            a.index_put_((indice1.float(),), values=self.to_device(val))
        true = torch.tensor(True, device="mlu")
        a = torch.randn(2, 3, device="mlu")
        a_expanded = a.expand(torch.Size([5, 1]) + a.size())
        ref_msg_3 = "shape mismatch: value tensor of shape [5, 1, 2, 3] cannot be broadcast to indexing result of shape [1, 2, 3]"
        with self.assertRaises(RuntimeError) as info:
            a[true] = a_expanded
        self.assertEqual(info.exception.args[0], ref_msg_3)
        a = torch.arange(0, 4, device="mlu")
        ref_msg_4 = "shape mismatch: value tensor of shape [0] cannot be broadcast to indexing result of shape [4]"
        with self.assertRaises(RuntimeError) as info:
            a[a > -1] = torch.tensor([]).mlu()
        self.assertEqual(info.exception.args[0], ref_msg_4)
        ref_msg_5 = "tensors used as indices must be long, byte or bool tensors"
        with self.assertRaises(IndexError) as info:
            a[torch.tensor([]).mlu()] = 1
        self.assertEqual(info.exception.args[0], ref_msg_5)
        # Remove the following test after problem in PYTORCH-12482 is addressed and solved
        x = torch.rand(4, 4, device="cpu")
        y = torch.rand(4, device="cpu")
        j = torch.tensor([2], device="mlu")
        ref_msg_6 = "Currently, index_put on MLU does not support input that is not on MLU, now the input is on cpu."
        with self.assertRaises(RuntimeError) as info:
            torch.ops.aten.index_put(x, j, y)
        self.assertEqual(info.exception.args[0], ref_msg_6)

    # @unittest.skip("not test")
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @testinfo()
    def test_index_put_bfloat16(self):
        param_list = [
            ((8, 8732, 4), [(8, 8732, 4)], torch.bool, 1, True),
            ((8, 8732, 4), [(8, 8732, 4)], torch.bool, 1, False),
        ]
        for input_shape, indices_shape, type_, value_, accumulate in param_list:
            input = torch.randn(input_shape, dtype=torch.bfloat16)
            input_cpu = input.float()
            input_cpu.requires_grad = True
            input_mlu = self.to_device(input)
            input_mlu.requires_grad = True
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                indice = self._generate_tensor(indices_shape[i], type_, input_shape[i])
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            output = torch.index_put(
                input_cpu, indices, torch.tensor(value_).float(), accumulate
            )
            output_mlu = torch.index_put(
                input_mlu,
                indices_mlu,
                self.to_device(torch.tensor(value_).bfloat16()),
                accumulate,
            )
            self.assertTensorsEqual(
                output.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
            )
            # test bfloat16 backward
            grad = torch.randn(output.shape, dtype=torch.bfloat16)
            grad_cpu = grad.float()
            output.backward(grad_cpu)
            grad_mlu = self.to_device(grad)
            output_mlu.backward(grad_mlu)
            self.assertTensorsEqual(
                input_cpu.grad, input_mlu.grad.cpu().float(), 0.003, use_MSE=True
            )

    # @unittest.skip("not test")
    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @largeTensorTest("24GB")
    def test_index_put_large(self):
        shape = [5, 1024, 1024, 1024]
        index = [1, 3, 4]
        x = torch.randn(shape).float()
        x_mlu = x.to("mlu")
        x_mlu_ptr = x_mlu.data_ptr()
        indices = torch.tensor(index).long()
        indices_mlu = indices.to("mlu")
        value = torch.randn(x[indices, :5].shape).float()
        value_mlu = value.to("mlu")
        x[indices, :5] = value
        x_mlu[indices_mlu, :5] = value_mlu
        self.assertTensorsEqual(x, x_mlu.cpu(), 0.00)
        self.assertEqual(x_mlu_ptr, x_mlu.data_ptr())

    @testinfo()
    @unittest.skipUnless(
        TEST_LARGETENSOR, "run largeTensorCases by `TEST_LARGETENSOR=TRUE` or `--large`"
    )
    @unittest.skipUnless(TEST_BFLOAT16, "Bfloat16 only support on MLU5xx")
    @largeTensorTest("46GB")
    def test_index_put_large_bfloat16(self):
        param_list = [
            ((5, 128, 1024, 1024), [(5, 128, 1024, 1024)], torch.bool, 1, True)
        ]
        for input_shape, indices_shape, type_, value_, accumulate in param_list:
            input = torch.randn(input_shape, dtype=torch.bfloat16)
            input_cpu = input.float()
            input_mlu = self.to_device(input)
            indices = []
            indices_mlu = []
            for i in range(len(indices_shape)):
                indice = self._generate_tensor(indices_shape[i], type_, input_shape[i])
                indices.append(indice)
                indices_mlu.append(self.to_device(indice))
            output = torch.index_put(
                input_cpu, indices, torch.tensor(value_).float(), accumulate
            )
            output_mlu = torch.index_put(
                input_mlu,
                indices_mlu,
                self.to_device(torch.tensor(value_).bfloat16()),
                accumulate,
            )
            self.assertTensorsEqual(
                output.float(), output_mlu.cpu().float(), 0.003, use_MSE=True
            )


if __name__ == "__main__":
    run_tests()
